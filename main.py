import torch
from dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from model import CellDSFormer, SVM
from torch import optim
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score, AP_score, AMI, ARI
import pickle

# 학습 설정 준비
def prepare():
    """
    학습 설정 및 데이터 준비
    :return: 학습 설정(args), K-Fold 데이터셋 리스트(trdt_list, tedt_list), 그래프 데이터(Gdata)
    """
    parser = argparse.ArgumentParser(description='AI4Bio')

    # 학습 파라미터 설정
    parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
    # parser.add_argument('--train_batch_size', type=int, default=12, help='')
    # parser.add_argument('--test_batch_size', type=int, default=24, help='')
    parser.add_argument('--train_batch_size', type=int, default=4, help='')
    parser.add_argument('--test_batch_size', type=int, default=8, help='')
    parser.add_argument('--data_path', type=str, default='./cell_dt', help='cell_dt is used for training')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu', help='')

    # parser.add_argument('--lr', type=int, default=0.0001, help='')
    parser.add_argument('--lr', type=int, default=0.001, help='')
    parser.add_argument('--folds', type=int, default=5, help='')
    parser.add_argument('--seq_length', type=int, default=2048, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--num_heads', type=int, default=8, help='')
    parser.add_argument('--num_heads_gat', type=int, default=3, help='')
    parser.add_argument('--d_ff', type=int, default=1024, help='')
    # parser.add_argument('--vocab_size', type=int, default=18628, help='')
    parser.add_argument('--vocab_size', type=int, default=26000, help='')
    parser.add_argument('--init_node_f', type=int, default=101, help='')
    parser.add_argument('--node_f', type=int, default=256, help='')

    args = parser.parse_args()

    # 그래프 데이터 로드
    with open('Gdata.pickle', 'rb') as file:
        Gdata = pickle.load(file)

    # Gdata['edge_index'] 유효성 검사 # 그래프 데이터 유효성 검사
    print("edge_index shape:", Gdata['edge_index'].shape)  # (2, num_edges) # 엣지 인덱스 모양 확인
    print("max edge_index:", Gdata['edge_index'].max())
    print("min edge_index:", Gdata['edge_index'].min())

    # 노드 수를 확인 후 유효성 검사 # 유효하지 않은 엣지 확인
    num_nodes = 18628 # 노드 개수
    assert Gdata['edge_index'].max() < num_nodes, "edge_index has values greater than num_nodes" "edge_index에 노드 개수를 초과하는 값이 있음"
    assert Gdata['edge_index'].min() >= 0, "edge_index has negative values" "edge_index에 음수 값이 있음"


    # K-Fold 데이터셋 생성
    trdt_list, tedt_list = KfoldDataset(args.data_path, args.folds)

    return args, trdt_list, tedt_list, Gdata

# 학습 및 검증 실행
def run():
    """
    학습 및 테스트 실행 함수
    """
    args, trdt_list, tedt_list, Gdata = prepare() # 설정 및 데이터 준비

    # 로그 파일 설정
    log_file_path = './model_save/training_log.txt'

    # 훈련 시작 전에 GPU 메모리 상태 확인
    print("Before training: ")
    print("Memory allocated:", torch.cuda.memory_allocated())
    print("Memory reserved:", torch.cuda.memory_reserved())

    # 그래프 데이터 디바이스로 이동
    Gdata = Gdata.to(args.device)

    # 각 Fold별로 학습 및 테스트 수행
    for f in [4, 3, 1, 2, 0]: # Fold 순서
        model = CellDSFormer(args) # 모델 초기화

        loss_function = torch.nn.CrossEntropyLoss() # 손실 함수 정의
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr) # 옵티마이저 설정

        model = model.to(args.device)

        # 학습 및 테스트 DataLoader 생성
        train_data_loader, test_data_loader = dataloader(current_fold=f, train_list=trdt_list, test_list=tedt_list,
                                                         tr_bs=args.train_batch_size, te_bs=args.test_batch_size)

        # 학습 루프
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
                    pred = model(input_ids, Gdata) # 모델 예측
                    labels = labels.float()
                    pred = pred.float()
                    loss = loss_function(pred, labels) # 손실 계산

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_sum = loss_sum + loss # 손실 합산
                    acc = Accuracy_score(pred, labels) # 정확도 계산
                    f1 = F1_score(pred, labels)
                    aur = AUROC_score(pred, labels)
                    pre = Precision_score(pred, labels)
                    rcl = Recall_score(pred, labels)
                    aps = AP_score(pred, labels)
                    ami = AMI(pred, labels)
                    ari = ARI(pred, labels)

                    Acc.append(acc)
                    F1.append(f1) # F1-score 계산
                    AUROC.append(aur) # AUROC 계산
                    Precision.append(pre) # 정밀도 계산
                    Recall.append(rcl) # 재현율 계산
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
                          'APscore:', sum(APscore) / len(APscore), 'Ami:',sum(Ami) / len(Ami), 'Ari:',sum(Ari)/ len(Ari))
                    
                    # 훈련 후 메모리 상태 확인 (옵션으로 추가 가능)
                    print("After training: ")
                    print("Memory allocated:", torch.cuda.memory_allocated())
                    print("Memory reserved:", torch.cuda.memory_reserved())

        # 모델 저장
        torch.save(model.state_dict(), './model_save/model.ckpt')

# 메인 실행
if __name__ == '__main__':
    run()
