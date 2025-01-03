import csv
import os
import pickle
from datasets import Dataset

# 유전자와 토큰 매핑 딕셔너리 생성 함수
def dictionary():
    """
    Collins_rCNV 데이터를 읽어 유전자 이름과 토큰 ID를 매핑하는 딕셔너리 생성
    :return: 유전자-토큰 딕셔너리
    """
    # 제외할 유전자 이름 목록
    d = ['1-Mar', 'MARCH8', 'MARCH5', '6-Mar', '3-Mar', 'MARCH1', '5-Mar', 'MARCH6', '2-Mar', '10-Mar', 'MARCH11', '11-Mar', '7-Mar',
         'MARCH9', 'MARCH4', '9-Mar', 'MARCH10', '15-Sep', 'MARCH7', '8-Mar', '4-Mar', 'MARCH3', '1-Dec', 'SEP15', 'MARCH2', 'DEC1']
    
    # Collins_rCNV 데이터 읽기
    with open('./Collins_rCNV_2022.dosage_sensitivity_scores/rCNV.gene_scores.tsv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = True
        i = 0
        token_gene_dictionary = {}
        for row in csv_reader:
            if header == True: # 첫 번째 행은 헤더로 무시
                header = False
                continue
            
            # 각 행에서 유전자 이름 추출
            row_data = row[0].split('\t')
            gene = row_data[0]
            
            # 제외 목록에 없는 유전자만 딕셔너리에 추가
            if gene not in d:
                token_gene_dictionary[gene] = i
                i = i + 1 # 다음 토큰 ID

    # 유전자-토큰 매핑 딕셔너리를 파일로 저장
    file_path = 'token_gene_dictionary.pickle'
    with open(file_path, 'wb') as file:
        pickle.dump(token_gene_dictionary, file)

    return token_gene_dictionary


# 유전자-토큰 매핑 딕셔너리 생성 및 로드
token_gene_dictionary = dictionary()
with open('./token_gene_dictionary.pickle', 'rb') as file:
    dictionary_genename_token_pair = pickle.load(file)

# 데이터 디렉토리 경로 설정
directory_path = './newdt'
csv.field_size_limit(500000) # CSV 필드 크기 제한 설정

# 데이터 전처리 함수
def rebuilder(directory_path):
    """
    디렉토리 내 CSV 파일을 읽고 전처리하여 Hugging Face Dataset으로 저장
    :param directory_path: 데이터가 저장된 디렉토리 경로
    """
    files_list = os.listdir(directory_path) # 디렉토리 내 파일 목록 가져오기
    labels_cell = [] # 레이블 리스트
    samples = [] # 샘플 리스트
    dt = {} # 최종 데이터셋을 저장할 딕셔너리

    # 각 파일 처리
    for filename in files_list:
        csv_file_path = directory_path + '/' + filename # 파일 경로 생성
        print(csv_file_path)

        headr = True # 헤더 처리 여부 플래그
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            pattern = [] # 유전자-토큰 매핑 리스트

            # 파일 읽기
            # sequence level
            for row in csv_reader:
                row_data = row[0].split('\t') # 탭으로 분리
                if headr:
                    # 헤더 행 처리: 유전자 이름을 토큰 ID로 매핑
                    for gene in row_data:
                        gene = gene.replace('"', '') # 따옴표 제거
                        if gene in dictionary_genename_token_pair:
                            pattern.append(dictionary_genename_token_pair[gene]) # 매핑된 토큰 추가
                        else:
                            pattern.append(-99999) # 매핑되지 않은 유전자는 -99999로 설정
                    headr = False
                    # print(pattern)
                else:
                    # 데이터 행 처리
                    assert len(pattern)==len(row_data) # 패턴과 데이터 길이가 동일한지 확인
                    seq_pattern_order_id_EXPscore = [] # 유전자-값 쌍 저장
                    # token level
                    for i in range(len(row_data)):
                        if i==0: # 첫 번째 열 무시
                            pass
                        elif i==1: # 두 번째 열에서 레이블 추출
                            if 'sensitive' in row_data[i]:
                                labels_cell.append(1)
                            elif 'resistant' in row_data[i]:
                                labels_cell.append(0)
                        else:
                            # 유효하지 않은 값 무시
                            if row_data[i]=='0':
                                pass
                            else:
                                if pattern[i]==-99999: # 매핑되지 않은 유전자 무시
                                    pass
                                else:
                                    seq_pattern_order_id_EXPscore.append((pattern[i],row_data[i]))

                    # EXPscore를 기준으로 정렬하고, 토큰만 추출
                    sorted_seq_pattern = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                    sample = [item[0] for item in sorted_seq_pattern]


                    # 최대 2048개의 토큰만 유지
                    while len(sample)<2048:
                        sample.append(0)  # 패딩 추가
                    assert len(sample)>=2048
                    sample =sample[:2048]

                    samples.append(sample)

        # 최종 데이터셋 구성
        dt['input_ids'] = samples
        dt['cell_label'] = labels_cell

        # Hugging Face Dataset으로 변환 후 저장
        my_dataset = Dataset.from_dict(dt)
        my_dataset.save_to_disk(filename[:-4] + '_dt')

# 데이터 전처리 실행
rebuilder(directory_path)




