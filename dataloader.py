from sklearn.model_selection import KFold

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import torch
from tqdm import tqdm
from datasets import load_from_disk

class BioDataset(Dataset):
    def __init__(self, f_path):

        """
        데이터셋 초기화: 저장된 데이터셋 파일을 로드하고, 토큰 및 라벨을 저장.
        :param f_path: 데이터셋 파일 경로
        """

        super(BioDataset, self).__init__()
        print('loading dataset...')
        # 파일에서 데이터셋 로드
        self.dataset = load_from_disk(f_path)

        # 입력 토큰과 라벨 로드
        # input_ids : 텍스트 데이터터를 토큰화한 결과. 
        self.tokens = self.dataset['input_ids']
        # cell_label : 각 샘플의 레이블
        self.labels = self.dataset['cell_label']
        # 데이터 길이 저장
        self.length = len(self.tokens)
        print('sequence number:',self.length)

    def __getitem__(self, item):
        """
        특정 인덱스의 데이터를 반환
        :param item: 데이터 인덱스
        :return: 입력 토큰과 라벨
        """
        return self.tokens[item], self.labels[item]

    def __len__(self):
        """
        데이터셋의 전체 길이 반환
        """
        return self.length


def bio_collate_fn(batches):
    """
    DataLoader에서 배치 데이터를 정리하는 함수.
    :param batches: 배치 데이터 리스트 [(tokens, label), ...]
    :return: 배치 토큰 텐서와 배치 라벨 텐서
    """
    batch_token = []
    batch_label = []
    for batch in batches:
        # 입력 토큰과 라벨을 리스트에 추가
        batch_token.append(torch.tensor(batch[0]))
        batch_label.append(torch.tensor(batch[1]))

    # 입력 토큰을 텐서로 스택
    batch_token = torch.stack(batch_token)

    # 라벨을 원-핫 벡터로 변환
    cell_one_hot_label = [torch.tensor([1 - item, item]) for item in batch_label]
    batch_label = torch.stack(cell_one_hot_label)


    return batch_token,batch_label

def KfoldDataset(train_data_folder, folds):
    """
    데이터셋을 K-fold 방식으로 분할
    :param train_data_folder: 데이터셋 경로
    :param folds: K값 (fold 개수)
    :return: K개의 학습 데이터 리스트, 테스트 데이터 리스트
    """

    # 데이터셋 로드
    biodataset = BioDataset(train_data_folder)
    # KFold 분할 초기화 (shuffle=True로 데이터 섞음)
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    trdt_list = [] # 학습 데이터 리스트
    tedt_list = [] # 테스트 데이터 리스트

    # 데이터셋을 K개의 fold로 분할
    for train_indices, val_indices in kf.split(biodataset):
        # 학습 및 테스트 데이터셋 생성
        train_dataset = torch.utils.data.Subset(biodataset, train_indices)
        test_dataset = torch.utils.data.Subset(biodataset, val_indices)
        trdt_list.append(train_dataset)
        tedt_list.append(test_dataset)


    return trdt_list, tedt_list




def dataloader(current_fold,train_list,test_list,tr_bs,te_bs):
    """
    현재 fold에 해당하는 학습 및 데스트 데이터 로더를 생성
    :param current_fold: 현재 처리 중인 fold 번호
    :param train_list: 학습 데이터 리스트
    :param test_list: 테스트 데이터 리스트
    :param tr_bs: 학습 배치 크기
    :param te_bs: 테스트 배치 크기
    :return: 학습 데이터 로더, 테스트 데이터 로더
    """

    # 학습 데이터 로더
    train_data_loader = DataLoader(dataset=train_list[current_fold], batch_size=tr_bs, shuffle=True,
                                   collate_fn=bio_collate_fn)
    # 테스트 데이터 로더
    test_data_loader = DataLoader(dataset=test_list[current_fold], batch_size=te_bs, shuffle=True,
                                  collate_fn=bio_collate_fn)

    return train_data_loader,test_data_loader





