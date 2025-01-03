""" import csv
import os
import pickle
from tqdm import tqdm

from datasets import Dataset
-

# with open('./token_gene_dictionary.pickle', 'rb') as file:
#    dictionary_genename_token_pair = pickle.load(file)

with open('dictionary_genename_token_pair.pickle', 'rb') as file:
    dictionary_genename_token_pair = pickle.load(file)

directory_path = './original_data'
csv.field_size_limit(10000000)

def rebuilder(directory_path):
    files_list = os.listdir(directory_path)
    labels_cell = []
    samples = []
    dt = {}

    # for filename in files_list:
    for filename in tqdm(files_list, desc="Processing files", ncols=100):
        csv_file_path = directory_path + '/' + filename
        print(csv_file_path)

        headr = True
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            pattern = []
            # sequence level
            for row in csv_reader:
                row_data = row[0].split('\t')
                if headr:
                    for gene in row_data:
                        gene = gene.replace('"', '')
                        if gene in dictionary_genename_token_pair:
                            pattern.append(dictionary_genename_token_pair[gene])
                        else:
                            pattern.append(-99999)
                    headr = False
                    # print(pattern)
                else:
                    assert len(pattern)==len(row_data)
                    seq_pattern_order_id_EXPscore = []
                    # token level
                    for i in range(len(row_data)):
                        if i==0:
                            pass
                        elif i==1:
                            if 'sensitive' in row_data[i]:
                                labels_cell.append(1)
                            elif 'resistant' in row_data[i]:
                                labels_cell.append(0)
                        else:
                            if row_data[i]=='0':
                                pass
                            else:
                                if pattern[i]==-99999: # none token
                                    pass
                                else:
                                    seq_pattern_order_id_EXPscore.append((pattern[i],row_data[i]))

                    sorted_seq_pattern = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                    sample = [item[0] for item in sorted_seq_pattern]


                    # only keep 2048 tokens
                    while len(sample)<2048:
                        sample.append(0)  # pad
                    assert len(sample)>=2048
                    sample =sample[:2048]

                    samples.append(sample)


    dt['input_ids'] = samples
    dt['cell_label'] = labels_cell


    my_dataset = Dataset.from_dict(dt)
    my_dataset.save_to_disk('./cell_dt')


rebuilder(directory_path)




 """

import csv
import os
import pickle
from tqdm import tqdm
from datasets import Dataset

# 1. 유전자 이름과 토큰 매핑 파일을 로드
with open('dictionary_genename_token_pair.pickle', 'rb') as file:
    dictionary_genename_token_pair = pickle.load(file)

# 2. 데이터가 저장된 디렉토리 경로
directory_path = './original_data'
csv.field_size_limit(10000000)  # 큰 파일을 처리하기 위해 필드 크기 제한 증가

# 데이터 재구성 함수
def rebuilder(directory_path):
    files_list = os.listdir(directory_path) # 디렉토리 내 모든 파일 목록 불러오기
    labels_cell = [] # 샘플 레이블 (0: resistant, 1: sensitive)
    samples = [] # 입력 샘플 (토큰 ID 리스트)
    dt = {} # 최종 데이터셋을 저장할 딕셔너리

    # 디렉토리 내 모든 파일을 처리
    for filename in tqdm(files_list, desc="Processing files", ncols=100):
        csv_file_path = os.path.join(directory_path, filename) # 파일 경로 생성
        print(f"Processing file: {csv_file_path}")

        headr = True # 첫 번째 행이 헤더인지 확인하는 플래그
        pattern = [] # 유전자 이름을 토큰으로 매핑한 리스트
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                row_data = row[0].split('\t') # 탭(`\t`)을 기준으로 행을 분리
                if headr:
                    # 3. 헤더 행 처리 (유전자 이름 -> 토큰 매핑)
                    for gene in row_data:
                        gene = gene.replace('"', '') # 유전자 이름의 따옴표 제거
                        if gene in dictionary_genename_token_pair:
                            pattern.append(dictionary_genename_token_pair[gene]) # 토큰 추가
                        else:
                            pattern.append(-99999) # 매핑되지 않은 유전자
                    headr = False # 헤더 처리 완료
                else:
                    # 4. 데이터 행 처리
                    seq_pattern_order_id_EXPscore = [] # 유효한 토큰-EXPscore 쌍 저장
                    for i in range(len(row_data)):
                        if i == 0:
                            pass  # 첫 번째 열은 무시
                        elif i == 1:
                            # 5. 레이블 생성 (sensitive: 1, resistant: 0)
                            if 'sensitive' in row_data[i]:
                                labels_cell.append(1)
                            elif 'resistant' in row_data[i]:
                                labels_cell.append(0)
                        else:
                            # 6. EXPscore가 0이 아니고 유효한 토큰일 경우 처리
                            if row_data[i] == '0':  # EXPscore가 0인 경우 스킵
                                pass
                            elif pattern[i] != -99999:  # 토큰이 유효하지 않은 경우 스킵
                                seq_pattern_order_id_EXPscore.append((pattern[i], row_data[i]))

                    # 7. EXPscore에 따라 토큰을 내림차순 정렬
                    sorted_seq_pattern = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                    sample = [item[0] for item in sorted_seq_pattern] # 정렬된 토큰 ID 추출

                    # 8. 최대 길이 2048로 자르고, 길이가 부족할 경우 패딩 추가
                    while len(sample) < 2048:
                        sample.append(0)  # 패딩 추가
                    assert len(sample) >= 2048
                    sample = sample[:2048]

                    samples.append(sample) # 최종 샘플 추가

    # 9. 데이터셋 구성 (입력 토큰과 레이블)
    dt['input_ids'] = samples
    dt['cell_label'] = labels_cell

    # 10. Hugging Face Dataset으로 변환 및 저장
    my_dataset = Dataset.from_dict(dt)
    my_dataset.save_to_disk('./cell_dt') # 데이터셋 디스크에 저장

rebuilder(directory_path) # 함수 실행
