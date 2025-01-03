import torch
from dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from model_sk import CellDSFormer, SVM
from torch import optim
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score, AP_score, AMI, ARI
import pickle

def prepare():
    parser = argparse.ArgumentParser(description='AI4Bio')
    parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--data_path', type=str, default='./new_dt', help='Path to training data')
    parser.add_argument('--device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu', help='Device to use')

    parser.add_argument('--lr', type=int, default=0.001, help='Learning rate')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--seq_length', type=int, default=2048, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_heads_gat', type=int, default=3, help='Number of heads for GAT')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward layer dimension')
    parser.add_argument('--vocab_size', type=int, default=26000, help='Vocabulary size')
    parser.add_argument('--init_node_f', type=int, default=101, help='Initial node feature dimension')
    parser.add_argument('--node_f', type=int, default=256, help='Node feature dimension')

    args = parser.parse_args()

    with open('Gdata.pickle', 'rb') as file:
        Gdata = pickle.load(file)

    # Gdata['edge_index'] 유효성 검사
    print("edge_index shape:", Gdata['edge_index'].shape)  # (2, num_edges)
    print("max edge_index:", Gdata['edge_index'].max())
    print("min edge_index:", Gdata['edge_index'].min())

    # 노드 수를 확인 후 유효성 검사
    num_nodes = 18628
    assert Gdata['edge_index'].max() < num_nodes, "edge_index has values greater than num_nodes"
    assert Gdata['edge_index'].min() >= 0, "edge_index has negative values"

    trdt_list, tedt_list = KfoldDataset(args.data_path, args.folds)

    return args, trdt_list, tedt_list, Gdata

def run():
    args, trdt_list, tedt_list, Gdata = prepare()

    # 로그 파일 설정
    log_file_path = './model_save/training_log_sk.txt'

    # 훈련 시작 전에 GPU 메모리 상태 확인
    print("Before training: ")
    print("Memory allocated:", torch.cuda.memory_allocated())
    print("Memory reserved:", torch.cuda.memory_reserved())

    Gdata = Gdata.to(args.device)
    for f in [4, 3, 1, 2, 0]:
        model = CellDSFormer(args)

        loss_function = torch.nn.CrossEntropyLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        model = model.to(args.device)

        train_data_loader, test_data_loader = dataloader(current_fold=f, train_list=trdt_list, test_list=tedt_list,
                                                         tr_bs=args.train_batch_size, te_bs=args.test_batch_size)

        for epoch in range(args.ep_num):
            loss_sum = 0
            Acc = []
            F1 = []
            AUROC = []
            Precision = []
            Recall = []
            APscore = []
            Ami = []
            Ari = []

            with tqdm(train_data_loader, ncols=100, position=0, leave=True, desc="Training epoch {}".format(epoch)) as batches:
                for b in batches:
                    input_ids, labels = b
                    # 입력 텐서의 최대값과 최소값 확인
                    print("input_ids shape:", input_ids.shape)
                    print("labels shape:", labels.shape)
                    print("max input_id:", input_ids.max())
                    print("min input_id:", input_ids.min())

                    # 디버깅 코드 추가
                    print(f"input_ids shape: {input_ids.shape}")
                    print(f"labels shape: {labels.shape}")

                    # CrossEntropyLoss의 레이블 값이 0 또는 1만 있는지 확인
                    print("labels unique values:", torch.unique(labels))

                    if torch.all(torch.logical_or(torch.all(labels == torch.tensor([1, 0])), torch.all(labels == torch.tensor([0, 1]))))==True:
                        continue
                    input_ids = input_ids.to(args.device)
                    labels = labels.to(args.device)
                    pred = model(input_ids, Gdata)

                    # 디버깅: 모델의 출력 텐서 크기와 값을 확인
                    print(f"Pred shape: {pred.shape}")
                    print(f"Pred values: {pred[:5]}")  # 처음 5개 예시 출력

                    labels = labels.float()
                    pred = pred.float()
                    loss = loss_function(pred, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_sum = loss_sum + loss
                    acc = Accuracy_score(pred, labels)
                    f1 = F1_score(pred, labels)
                    aur = AUROC_score(pred, labels)
                    pre = Precision_score(pred, labels)
                    rcl = Recall_score(pred, labels)
                    aps = AP_score(pred, labels)
                    ami = AMI(pred, labels)
                    ari = ARI(pred, labels)

                    Acc.append(acc)
                    F1.append(f1)
                    AUROC.append(aur)
                    Precision.append(pre)
                    Recall.append(rcl)
                    APscore.append(aps)
                    Ami.append(ami)
                    Ari.append(ari)

                # 파일에 훈련 결과 기록
                with open(log_file_path, 'a') as f:
                    f.write(f'Training epoch: {epoch}, Current_fold: {f}, loss: {loss_sum}, '
                            f'Accuracy: {sum(Acc) / len(Acc)}, AUROC: {sum(AUROC) / len(AUROC)}, '
                            f'Precision: {sum(Precision) / len(Precision)}, Recall: {sum(Recall) / len(Recall)}, '
                            f'F1: {sum(F1) / len(F1)}, APscore: {sum(APscore) / len(APscore)}, '
                            f'Ami: {sum(Ami) / len(Ami)}, Ari: {sum(Ari) / len(Ari)}\n')

                print('Training epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:', sum(Acc) / len(Acc),
                      'AUROC:', sum(AUROC) / len(AUROC),
                      'Precision:', sum(Precision) / len(Precision), 'Recall:', sum(Recall) / len(Recall), 'F1:',
                      sum(F1) / len(F1),
                      'APscore:', sum(APscore) / len(APscore), 'Ami:', sum(Ami) / len(Ami), 'Ari:', sum(Ari) / len(Ari))

            loss_sum = 0
            Acc = []
            F1 = []
            AUROC = []
            Precision = []
            Recall = []
            with torch.no_grad():
                with tqdm(test_data_loader, ncols=100, position=0, leave=True, desc="Testing epoch {}".format(epoch)) as batches:
                    for b in batches:
                        input_ids, labels = b
                        if torch.all(torch.logical_or(torch.all(labels == torch.tensor([1, 0])),
                                                      torch.all(labels == torch.tensor([0, 1])))) == True:
                            continue
                        input_ids = input_ids.to(args.device)
                        labels = labels.to(args.device)
                        pred = model(input_ids, Gdata)

                        # 디버깅: 모델의 출력 텐서 크기와 값을 확인
                        print(f"Pred shape: {pred.shape}")
                        print(f"Pred values: {pred[:5]}")  # 처음 5개 예시 출력

                        labels = labels.float()
                        pred = pred.float()
                        loss = loss_function(pred, labels)

                        loss_sum = loss_sum + loss
                        acc = Accuracy_score(pred, labels)
                        f1 = F1_score(pred, labels)
                        aur = AUROC_score(pred, labels)
                        pre = Precision_score(pred, labels)
                        rcl = Recall_score(pred, labels)
                        aps = AP_score(pred, labels)
                        ami = AMI(pred, labels)
                        ari = ARI(pred, labels)

                        Acc.append(acc)
                        F1.append(f1)
                        AUROC.append(aur)
                        Precision.append(pre)
                        Recall.append(rcl)
                        APscore.append(aps)
                        Ami.append(ami)
                        Ari.append(ari)

                    # 파일에 테스트 결과 기록
                    with open(log_file_path, 'a') as f:
                        f.write(f'Testing epoch: {epoch}, Current_fold: {f}, loss: {loss_sum}, '
                                f'Accuracy: {sum(Acc) / len(Acc)}, AUROC: {sum(AUROC) / len(AUROC)}, '
                                f'Precision: {sum(Precision) / len(Precision)}, Recall: {sum(Recall) / len(Recall)}, '
                                f'F1: {sum(F1) / len(F1)}, APscore: {sum(APscore) / len(APscore)}, '
                                f'Ami: {sum(Ami) / len(Ami)}, Ari: {sum(Ari) / len(Ari)}\n')

                    print('Testing epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:',
                          sum(Acc) / len(Acc), 'AUROC:', sum(AUROC) / len(AUROC),
                          'Precision:', sum(Precision) / len(Precision), 'Recall:', sum(Recall) / len(Recall), 'F1:',
                          sum(F1) / len(F1),
                          'APscore:', sum(APscore) / len(APscore), 'Ami:', sum(Ami) / len(Ami), 'Ari:', sum(Ari) / len(Ari))

                    # 훈련 후 메모리 상태 확인 (옵션으로 추가 가능)
                    print("After training: ")
                    print("Memory allocated:", torch.cuda.memory_allocated())
                    print("Memory reserved:", torch.cuda.memory_reserved())

        torch.save(model.state_dict(), './model_save/model_sk.ckpt')

if __name__ == '__main__':
    run()