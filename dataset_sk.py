import os
import csv
import pickle
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import re  # 정규식 모듈 추가

# 1. 기존의 dictionary(pickle 파일) 로드
with open('dictionary_genename_token_pair.pickle', 'rb') as file:
    dictionary_genename_token_pair = pickle.load(file)

# 2. scdata (Single-cell gene expression data) 및 label (Drug response label) 로드
scdata_file = '/data/project/eunyi/DrugFormer/data_sk/data_sk_singlecell.csv'  # scdata 파일 경로
label_file = '/data/project/eunyi/DrugFormer/data_sk/data_sk_druglabel.csv'  # label 파일 경로

scdata_df = pd.read_csv(scdata_file)
label_df = pd.read_csv(label_file)

# 3. 유전자 이름을 정규화하여 하이픈(-)과 점(.)을 동일하게 처리
def normalize_gene_name_v2(gene_name):
    return re.sub(r'\.', '-', gene_name)  # '.'을 '-'로 바꾸기

# 데이터프레임과 토큰 딕셔너리에서 유전자 이름을 정규화
genes_in_scdata_normalized_v2 = [normalize_gene_name_v2(gene) for gene in scdata_df.columns[1:]]  # scdata의 유전자 목록 정규화
genes_in_token_dict_normalized_v2 = {normalize_gene_name_v2(gene): token for gene, token in dictionary_genename_token_pair.items()}  # dictionary의 유전자 정규화

# 4. 데이터에 존재하지 않는 유전자들 확인 (정규화된 이름을 사용)
missing_genes = [gene for gene in genes_in_scdata_normalized_v2 if gene not in genes_in_token_dict_normalized_v2]

if missing_genes:
    print(f"다음 유전자들은 dictionary_genename_token_pair에 없습니다: {missing_genes}")

# 5. 데이터 준비: 유전자 리스트를 토큰화
samples = []
labels_cell = []

# tqdm을 사용하여 진행 상태 표시
for idx, row in tqdm(scdata_df.iterrows(), total=scdata_df.shape[0], desc="Processing scdata", ncols=100):
    sample = []
    for gene in genes_in_scdata_normalized_v2:
        if gene in genes_in_token_dict_normalized_v2:
            sample.append(genes_in_token_dict_normalized_v2[gene])
        else:
            sample.append(-99999)  # 존재하지 않는 유전자는 -99999로 표시

    samples.append(sample)
    # label 정보 추가
    if label_df.loc[idx, 'label'] == 'sensitive':
        labels_cell.append(1)
    elif label_df.loc[idx, 'label'] == 'resistant':
        labels_cell.append(0)

# 6. 데이터 길이 조정 (패딩 및 자르기)
max_seq_len = 2048
for i in tqdm(range(len(samples)), desc="Padding and trimming samples", ncols=100):
    sample = samples[i]
    while len(sample) < max_seq_len:
        sample.append(0)  # Padding
    sample = sample[:max_seq_len]  # 2048개로 자르기
    samples[i] = sample

# 7. Dataset 준비
dt = {}
dt['input_ids'] = samples
dt['cell_label'] = labels_cell

# 8. Dataset 객체로 변환
my_dataset = Dataset.from_dict(dt)

# 9. Dataset을 disk에 저장
my_dataset.save_to_disk('./new_dt')

print("Dataset 생성 완료 및 './new_dt' 폴더에 저장되었습니다.")