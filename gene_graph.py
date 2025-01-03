import csv
import os
import pickle
import torch
from datasets import Dataset
from datasets import load_from_disk
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# 유전자와 토큰 매핑 생성
def dictionary():
    """
    유전자 이름과 토큰(숫자 ID) 매핑 딕셔너리를 생성
    :return: 유전자-토큰 딕셔너리
    """
    # 제외할 유전자 이름 목록
    d = ['1-Mar', 'MARCH8', 'MARCH5', '6-Mar', '3-Mar', 'MARCH1', '5-Mar', 'MARCH6', '2-Mar', '10-Mar', 'MARCH11', '11-Mar', '7-Mar',
         'MARCH9', 'MARCH4', '9-Mar', 'MARCH10', '15-Sep', 'MARCH7', '8-Mar', '4-Mar', 'MARCH3', '1-Dec', 'SEP15', 'MARCH2', 'DEC1'] 
    with open('./Collins_rCNV_2022.dosage_sensitivity_scores/rCNV.gene_scores.tsv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = True
        i = 0
        token_gene_dictionary = {}
        for row in csv_reader:
            if header == True: # 헤더 행을 무시
                header = False
                continue
            row_data = row[0].split('\t')
            gene = row_data[0] # 유전자 이름
            if gene not in d: # 제외 목록에 없는 유전자만 추가
                token_gene_dictionary[gene] = i # 유전자 이름에 토큰 ID 할당
                i = i + 1 # 다음 토큰 ID


    # 딕셔너리를 pickle 파일로 저장
    file_path = 'token_gene_dictionary.pickle'
    with open(file_path, 'wb') as file:
        pickle.dump(token_gene_dictionary, file)

    return token_gene_dictionary


# 유전자 간 평균 점수를 기반으로 엣지 생성
def dictionary_meanScores_token_pair_building(dictionary):
    """
    유전자 간의 평균 점수를 기반으로 엣지(Edge) 정보를 생성
    :param dictionary: 유전자-토큰 딕셔너리
    :return: 엣지 인덱스와 엣지 가중치
    """
    dictionary_meanScores_token_pair = {}
    with open('./Collins_rCNV_2022.dosage_sensitivity_scores/rCNV.gene_scores.tsv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = True
        for row in csv_reader:
            if header == True: # 헤더 행을 무시
                header = False
                continue
            row_data = row[0].split('\t')
            gene = row_data[0]
            if gene in dictionary: # 유전자 이름이 딕셔너리에 있을 경우
                score1 = float(row_data[1]) # 첫 번째 점수
                score2= float(row_data[2]) # 두 번째 점수
                mean = (score1 + score2) / 2 # 평균 점수 계산
                dictionary_meanScores_token_pair[dictionary[gene]] = mean

    # 엣지 정보 생성
    l_score = []
    for i in range(len(dictionary_meanScores_token_pair)):
        l_score.append(dictionary_meanScores_token_pair[i])

    edge_index = []
    edge_weight = []
    for i in range(len(l_score)):
        for j in range(i,len(l_score)):
            edge_value = 1 - abs(l_score[i] - l_score[j]) # 유사도 계산
            if edge_value>0.999: # 임계값 이상의 유사도만 엣지로 추가
                edge_index.append([i,j])
                edge_weight.append(edge_value)


    edge_index = torch.tensor(edge_index).t().contiguous() # 엣지 인덱스를 PyTorch 텐서로 변환
    edge_weight = torch.tensor(edge_weight) # 엣지 가중치를 PyTorch 텐서로 변환
    return edge_index, edge_weight


# 유전자 노드 특징 생성
def dictionary_eigenfeatures_token_pair_building(dictionary):
    """
    유전자 노드의 특징 벡터를 생성
    :param dictionary: 유전자-토큰 딕셔너리
    :return: 노드 특징 벡터
    """
    dictionary_eigenfeatures_token_pair = {}
    with open('./Collins_rCNV_2022.gene_features_matrix/Collins_rCNV_2022.gene_eigenfeatures1.csv', 'r') as file:
        csv_reader = csv.reader(file)
        header = True
        for row in csv_reader:
            if header == True: # 헤더 행을 무시
                header = False
                continue

            gene = row[3] # 유전자 이름
            if gene in dictionary:
                f = []
                for i in range(4,len(row)): # 특징 벡터
                    f.append(float(row[i]))
                dictionary_eigenfeatures_token_pair[dictionary[gene]] = f


    # 특징 벡터를 리스트로 정렬
    l = []
    for i in range(len(dictionary_eigenfeatures_token_pair)):
        l.append(dictionary_eigenfeatures_token_pair[i])


    node_features = torch.tensor(l) # PyTorch 텐서로 변환
    return node_features


# 메인 실행 부분
token_gene_dictionary = dictionary() # 유전자-토큰 매핑 생성
edge_index, edge_weights = dictionary_meanScores_token_pair_building(token_gene_dictionary)
print(edge_index.shape)
print(edge_weights.shape)

node_features = dictionary_eigenfeatures_token_pair_building(token_gene_dictionary) # 노드 특징 생성
print(node_features.shape) 

# 그래프 데이터 객체 생성
Gdata = Data(x=node_features, edge_index=edge_index, edge_weights=edge_weights)

# 그래프 데이터 저장
file_path = 'Gdata.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(Gdata, file)
