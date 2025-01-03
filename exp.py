from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import csv
from dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from model import CellDSFormer
from torch import optim
from sklearn.metrics import accuracy_score, f1_score

import pickle
import torch
import torch.nn.functional as F

# 데이터셋 클래스를 정의 클래스
class BioDataset(Dataset):
    def __init__(self, f_path):
        """
        데이터셋 초기화: 저장된 데이터셋 파일을 로드하고, 토큰 및 레이블을 저장합니다.
        :param f_path: 데이터셋 파일 경로
        """
        super(BioDataset, self).__init__()
        print('loading dataset...')
        self.dataset = load_from_disk(f_path) # Hugging Face 데이터셋 로드

        # 입력 토큰과 라벨 초기화
        self.tokens = self.dataset['input_ids']
        self.labels = self.dataset['cell_label']


        self.length = len(self.tokens)

    def __getitem__(self, item):
        """
        특정 인덱스의 데이터를 반환
        :param item: 데이터 인덱스
        :return: 입력 토큰과 라벨
        """
        return self.tokens[item], self.labels[item]

    def __len__(self):
        """
        데이터셋의 전체 길이를 반환
        """
        return self.length

# 배치 데이터를 처리하는 함수 정의
def bio_collate_fn(batches):
    """
    DataLoader에서 배치 데이터를 정리하는 함수
    :param batches: 배치 데이터 리스트 [(tokens, label), ...]
    :return: 배치 토큰 텐서와 배치 라벨 텐서
    """
    batch_token = []
    batch_label = []
    for batch in batches:
        batch_token.append(torch.tensor(batch[0])) # 토큰 추가
        batch_label.append(torch.tensor(batch[1])) # 라벨 추가

    batch_token = torch.stack(batch_token) # 토큰 텐서로 스택
    batch_label = torch.stack(batch_label) # 라벨 텐서로 스택



    return batch_token,batch_label






# 실험 설정 및 파라미터 준비
def prepare():
    """
    argparse를 통해 실험 설정을 정의하고, 사용자 입력을 처리
    """
    parser = argparse.ArgumentParser(description='AI4Bio')

    # 학습 파라미터
    parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
    parser.add_argument('--train_batch_size', type=int, default=32, help='')
    parser.add_argument('--test_batch_size', type=int, default=64, help='')
    parser.add_argument('--data_path', type=str, default='./GSE207422_Tor_post_dt', help='')   # newdt
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')
    #parser.add_argument('--device', type=str, default='cpu', help='')

    parser.add_argument('--seq_length', type=int, default=2048, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--num_heads', type=int, default=8, help='')
    parser.add_argument('--num_heads_gat', type=int, default=3, help='')
    parser.add_argument('--d_ff', type=int, default=1024, help='')
    parser.add_argument('--vocab_size', type=int, default=18628, help='')
    parser.add_argument('--init_node_f', type=int, default=101, help='')
    parser.add_argument('--node_f', type=int, default=256, help='')

    args = parser.parse_args()


    return args

# F1-score 계산 함수
def F1_score(pred, labels):
    """
    F1-score 계산
    :param pred: 모델 예측값
    :param labels: 실제 라벨
    :return: F1-score
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값의 최대 확률 인덱스
    max_prob_index_labels = labels.cpu() # 실제 라벨
    F1 = f1_score(max_prob_index_pred, max_prob_index_labels) # F1-score 계산

    return F1

# 실험 실행 함수
def run():
    args= prepare() # 설정 준비

    # 테스트 데이터셋 준비
    biodataset = BioDataset('./GSE162117_mal_pre_countsMatrix_dt')  #GSE162117_mal_pre_countsMatrix_dt   GSE161801_IMiD_mal_pre_countsMatrix_dt
    test_data_loader = DataLoader(dataset=biodataset, batch_size=64, shuffle=True,
                                  collate_fn=bio_collate_fn)

    # 모델 초기화 및 로드
    model = CellDSFormer(args)

    model.load_state_dict(torch.load('./model_save/model.ckpt')) # 저장된 가중치 로드
    model.eval() # 평가 모드로 설정
    model = model.to(args.device)

    F1 = [] # F1-score 저장
    out = [] # 모델 예측 결과 저장

    # 그래프 데이터 로드
    with open('Gdata.pickle', 'rb') as file:
        Gdata = pickle.load(file)
    Gdata = Gdata.to(args.device)

    # 모델 평가
    with (torch.no_grad()):
        with tqdm(test_data_loader, ncols=80, position=0, leave=True) as batches:
            for b in batches:  # 배치 처리
                input_ids, labels = b  # 입력 토큰과 라벨
                input_ids = input_ids.to(args.device)
                labels = labels.to(args.device)
                preds = model(input_ids, Gdata) # 모델 예측
                f1 = F1_score(preds.float(), labels.float()) # F1-score 계산
                F1.append(f1)
                preds = torch.sigmoid(preds) # 시그모이드 활성화 함수 적용

                preds = preds.tolist() # 리스트로 변환

                for sample_results in preds:
                    out.append(sample_results)

    # F1-score 출력
    print('F1:',sum(F1) / len(F1))
    # csv_file_path = './GSE161801_IMiD_mal_pre_countsMatrix_dt.csv'
    #
    # with open(csv_file_path, mode='w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerows(out)


# 실행
run()
