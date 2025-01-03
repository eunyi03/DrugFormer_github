from torch import nn
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F

# CellDSFormer 클래스 정의
class CellDSFormer(nn.Module):
    def __init__(self, args):
        super(CellDSFormer, self).__init__()
        self.args = args

        # 단어 및 위치 임베딩
        # 단어 임베딩 레이어: 각 단어를 d_model 차원의 벡터로 임베딩
        self.word_embed_layer = nn.Embedding(self.args.vocab_size, self.args.d_model)       
        # 위치 임베딩 레이어: 시퀀스 내 위치를 d_model 차원의 벡터로 임베딩
        self.pos_embed_layer = nn.Embedding(self.args.seq_length, self.args.d_model)

        # Graph Attention Network(GAT) 레이어
        # 그래프 신경망 레이어: GAT1과 GAT2는 그래프 데이터를 처리하기 위한 GATConv 레이어
        self.GAT1 = GAT(in_dims=self.args.init_node_f, out_dims=self.args.d_model, num_heads=self.args.num_heads_gat)
        self.GAT2 = GAT(in_dims=self.args.node_f, out_dims=self.args.d_model, num_heads=self.args.num_heads_gat)

        # Transformer Encoder 레이어 6개
        # 여러 Transformer Encoder Layer 정의: 각 Transformer Layer는 입력을 처리하고, 특성 추출
        self.Transformer1 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer2 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer3 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer4 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer5 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer6 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)

        # Aggregator 레이어 (결합된 텐서를 처리)
        # 두 개의 Aggregator 레이어: 두 텐서를 결합한 후, 정보를 압축하고 비선형 활성화
        self.aggregator1 = nn.Sequential(
            nn.Linear(self.args.d_model * 2, self.args.d_model),
            nn.ReLU()
            )
        self.aggregator2 = nn.Sequential(
            nn.Linear(self.args.d_model * 2, self.args.d_model),
            nn.ReLU()
            )

        # Bottleneck 및 최종 분류 레이어
        # Bottleneck 레이어: 차원 축소를 위한 선형 변환
        self.bottleneck_layer = nn.Linear(self.args.seq_length * self.args.d_model, 32)
        # 최종 출력 레이어: 이진 분류를 위한 선형 레이어
        self.classfier = nn.Linear(32, 2)


    # GFcandidate 함수: 입력된 시퀀스와 그래프 특성에 맞춰 선택된 그래프 특성 값 반환
    def GFcandidate(self, seq, gf):
        seq = seq.long()  # 입력 시퀀스를 long 타입으로 변환
        seq_gf_pair = gf[seq]  # seq에 해당하는 그래프 특성 추출

        return seq_gf_pair


    # 모델의 forward 함수 정의
    def forward(self, seq, Gdata):
        # 위치 임베딩 정의
        seq_pos = torch.arange(2048)  # 0부터 2047까지 숫자 생성 (시퀀스 길이)
        seq_pos = seq_pos.reshape(1, 2048).to(self.args.device)  # 디바이스에 맞춰 위치 텐서 생성
        
        # 단어 임베딩과 위치 임베딩 결합
        seq_word_embed = self.word_embed_layer(seq)  # 시퀀스에 대해 단어 임베딩
        seq_pos_embed = self.pos_embed_layer(seq_pos)  # 위치에 대한 임베딩
        embed = seq_word_embed + seq_pos_embed  # 단어와 위치 임베딩을 더함

        # -9999 값을 가진 위치는 0으로 처리
        embed = torch.where(seq.unsqueeze(-1) == -9999, torch.zeros_like(embed), embed)

        # Transformer1 적용
        embed1 = self.Transformer1(embed)

        # GAT1을 이용해 그래프 특성 추출
        graphf1 = self.GAT1(Gdata['x'], Gdata['edge_index'], Gdata['edge_weights'])

        # seq와 graphf1의 크기와 범위가 맞는지 확인
        print(f"Max index in seq: {seq.max()}, Min index in seq: {seq.min()}")
        print(f"Graph shape: {graphf1.shape}")

        # seq가 graphf1의 범위 내에 있는지 확인, 범위를 벗어난 값은 클램프 처리
        seq = torch.clamp(seq, max=graphf1.size(0) - 1)  # seq의 값이 graphf1 크기를 초과하지 않도록 수정
        print(f"Clamped seq: {seq[:5]}")  # 클램프된 값 확인 (디버깅)

        # 그래프 특성에 기반하여 후보 데이터 추출
        seq_gf_pair1 = self.GFcandidate(seq, graphf1)

        # 임베딩과 그래프 후보 데이터를 결합
        fused_embed1 = torch.cat((embed1, seq_gf_pair1), dim=2)

        # 결합된 임베딩을 aggregator1에 통과시켜 정보 추출
        fused_embed1 = self.aggregator1(fused_embed1)

        # Transformer layers 처리
        embed2 = self.Transformer2(fused_embed1)
        embed3 = self.Transformer3(embed2)
        embed4 = self.Transformer4(embed3)
        embed5 = self.Transformer5(embed4)
        embed6 = self.Transformer6(embed5)

        # GAT2를 이용해 다시 그래프 특성 추출
        graphf2 = self.GAT2(graphf1, Gdata['edge_index'], Gdata['edge_weights'])
        seq_gf_pair2 = self.GFcandidate(seq, graphf2)

        # 두 번째 임베딩과 그래프 후보 데이터를 결합
        fused_embed2 = torch.cat((embed6, seq_gf_pair2), dim=2)
        fused_embed2 = self.aggregator2(fused_embed2)

        # 최종 결과를 위해 임베딩을 펼침
        x = fused_embed2.view(fused_embed2.size(0), -1)

        # 차원 축소를 위한 bottleneck 레이어
        x = self.bottleneck_layer(x)

        # 최종 출력 레이어 (이진 분류)
        x = self.classfier(x)

        return x


# GAT (Graph Attention Network) 모델 정의
class GAT(torch.nn.Module):
    def __init__(self, in_dims, out_dims, num_heads):
        super(GAT, self).__init__()
        self.gat = GATConv(in_dims, out_dims, heads=num_heads)  # GATConv 레이어 정의
        self.l = nn.Linear(out_dims * num_heads, out_dims)  # GAT 출력 차원에 맞는 선형 레이어

    def forward(self, x, edge_index, edge_weights):
        x = self.gat(x, edge_index, edge_weights)  # GATConv을 적용
        x = F.relu(x)  # 활성화 함수 적용
        x = F.dropout(x, p=0.1)  # 드롭아웃 적용
        x = self.l(x)  # 선형 변환 적용
        return x


# SVM 모델 정의 (단순 선형 분류기)
class SVM(nn.Module):
    def __init__(self, args):
        super(SVM, self).__init__()
        self.input_size = args.seq_length  # 시퀀스 길이를 입력 크기로 설정
        self.fc = nn.Linear(self.input_size, 2)  # 이진 분류를 위한 선형 레이어

    def forward(self, seq, Gdata):
        output = self.fc(seq.float())  # 시퀀스를 선형 변환
        return output