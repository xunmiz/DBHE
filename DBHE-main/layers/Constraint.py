import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. [新增] 适用于 GNN Expert Embeddings 的正交约束
# ==========================================
class OrthoLoss(nn.Module):
    '''
    用于 GNN 超网络生成的专家基 (Expert Embeddings) 的正交约束。
    替代了原有的 D_constraint1 和 D_constraint2。
    
    目标：
    1. 归一化：每个专家 Embedding 模长接近 1。
    2. 正交化：不同专家 Embedding 之间的余弦相似度接近 0 (互相排斥，保证多样性)。
    '''
    def __init__(self, strength=1e-3):
        super(OrthoLoss, self).__init__()
        self.strength = strength

    def forward(self, expert_embeddings):
        """
        输入: expert_embeddings [Num_Experts, Graph_Dim]
        """
        # 1. 归一化 (Normalize)
        # 使得每个向量模长为 1，这样矩阵乘法结果即为余弦相似度
        # 加上 1e-8 防止除零
        norms = torch.norm(expert_embeddings, p=2, dim=1, keepdim=True)
        normalized_emb = expert_embeddings / (norms + 1e-8)

        # 2. 计算 Gram Matrix (相似度矩阵)
        # [N, D] @ [D, N] -> [N, N]
        # sim_matrix[i][j] 表示专家 i 和专家 j 的相似度
        sim_matrix = torch.matmul(normalized_emb, normalized_emb.T)

        # 3. 构建目标单位矩阵 (Identity Matrix)
        # 我们希望对角线为 1 (自身模长)，非对角线为 0 (相互正交)
        identity = torch.eye(sim_matrix.size(0), device=expert_embeddings.device)

        # 4. 计算 Frobenius Norm Loss (MSE 的变体)
        # 计算 (sim_matrix - I) 的 F-范数的平方
        loss = torch.norm(sim_matrix - identity, p='fro') ** 2
        
        return self.strength * loss


# ==========================================
# 2. [保留] 原有的约束 (Legacy)
# ==========================================
class D_constraint1(torch.nn.Module):
    '''
    (原代码) 该约束确保矩阵 d 的列近似正交，且每列向量的长度接近1。即使得 d 近似于正交矩阵。
    通常用于子空间聚类中的投影矩阵。
    '''
    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        I = torch.eye(d.shape[1]).to(d.device)  # .cuda()
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return 1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):
    '''
    (原代码) 该约束确保不同聚类之间的矩阵 d 的子块尽量独立。
    d.shape: [Feature_Dim, n_clusters * sub_dim]
    用于约束不同子空间之间的正交性。
    '''
    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim, n_clusters):
        S = torch.ones(d.shape[1], d.shape[1]).to(d.device)
        zero = torch.zeros(dim, dim)
        
        # 将矩阵 S 按块分为 n_clusters 个大小为 dim×dim 的子矩阵，并将这些子矩阵设为 zero。
        # 目的是只惩罚不同聚类之间的相关性 (非对角块)，忽略聚类内部的相关性 (对角块)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
            
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return 1e-3 * loss_d2_constraint