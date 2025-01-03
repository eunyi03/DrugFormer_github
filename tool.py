from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

# 정확도 (Accuracy) 계산 함수
def Accuracy_score(pred, labels):
    """
    정확도를 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: 정확도 (float)
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값에서 최대 확률 인덱스 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블의 최대값 인덱스 추출
    acc = accuracy_score(max_prob_index_labels, max_prob_index_pred) # 정확도 계산

    return acc

# F1-score 계산 함수
def F1_score(pred, labels):
    """
    F1-score를 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: F1-score (float)
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값에서 최대 확률 인덱스 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블의 최대값 인덱스 추출
    F1 = f1_score(max_prob_index_pred, max_prob_index_labels) # F1-score 계산

    return F1

# AUROC (Area Under ROC Curve) 계산 함수
def AUROC_score(pred, labels):
    """
    AUROC를 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: AUROC (float)
    """
    max_prob_index_pred = pred[:, 1].view(-1, 1).cpu().detach().numpy() # 클래스 1의 확률값 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블
    AUROC = roc_auc_score(max_prob_index_labels, max_prob_index_pred) # AUROC 계산

    return AUROC

# 정밀도 (Precision) 계산 함수
def Precision_score(pred, labels):
    """
    정밀도를 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: 정밀도 (float)
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값에서 최대 확률 인덱스 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블의 최대값 인덱스 추출
    pre = precision_score(max_prob_index_labels, max_prob_index_pred,zero_division=1) # 정밀도 계산

    return pre

# 재현율 (Recall) 계산 함수
def Recall_score(pred, labels):
    """
    재현율을 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: 재현율 (float)
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값에서 최대 확률 인덱스 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블의 최대값 인덱스 추출
    rcl = recall_score(max_prob_index_labels, max_prob_index_pred,zero_division=1) # 재현율 계산

    return rcl

# Average Precision (AP) 계산 함수
def AP_score(pred, labels):
    """
    Average Precision Score를 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: Average Precision Score (float)
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값에서 최대 확률 인덱스 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블의 최대값 인덱스 추출
    aps = average_precision_score(max_prob_index_labels, max_prob_index_pred) # AP 계산

    return aps

# Adjusted Mutual Information (AMI) 계산 함수
def AMI(pred, labels):
    """
    Adjusted Mutual Information (AMI)를 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: Adjusted Mutual Information (float)
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값에서 최대 확률 인덱스 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블의 최대값 인덱스 추출
    ami = adjusted_mutual_info_score(max_prob_index_labels, max_prob_index_pred) # AMI 계산

    return ami

# Adjusted Rand Index (ARI) 계산 함수
def ARI(pred, labels):
    """
    Adjusted Rand Index (ARI)를 계산
    :param pred: 모델의 예측값 (logits)
    :param labels: 실제 레이블 (원-핫 인코딩)
    :return: Adjusted Rand Index (float)
    """
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu() # 예측값에서 최대 확률 인덱스 추출
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu() # 실제 레이블의 최대값 인덱스 추출
    ari = adjusted_rand_score(max_prob_index_labels, max_prob_index_pred) # ARI 계산

    return ari