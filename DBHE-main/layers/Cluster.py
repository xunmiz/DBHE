import torch
import torch.nn as nn
import torch.nn.functional as F

class EDESC(nn.Module):
    def __init__(self,
                 d_model,
                 n_clusters,
                 eta,
                 c_out,
                 bs,
                 patch_len,
                 stride):
        super(EDESC, self).__init__()
        self.n_clusters = n_clusters
        self.eta = eta
        self.c_out = c_out
        self.bs = bs
        self.patch_len = patch_len
        self.stride = stride

        # [修改] 移除 self.D 的定义
        # 原代码: self.D = Parameter(torch.Tensor(n_clusters * self.d, n_clusters * self.d))
        # 现在聚类中心由 GNN Hypernetwork 动态生成并传入
        
        # [保留] 用于辅助计算或作为元数据 (可选)
        n_z = c_out * d_model
        self.d = int(n_z / n_clusters)

    def reverse_unfold(self, z, original_length, stride):
        """
        工具函数：用于重构时间序列，保持原逻辑不变
        z: [bs x patch_num x nvars x patch_len]
        """
        bs, patch_num, nvars, patch_len = z.size()
        output = torch.zeros((bs, nvars, original_length), device=z.device)
        patch_counts = torch.zeros((bs, nvars, original_length), device=z.device)

        for i in range(patch_num):
            start = i * stride
            end = start + patch_len
            if end > original_length:
                end = original_length

            # Dynamically adjust the patch length if it exceeds the original length
            current_patch_len = end - start

            output[:, :, start:end] += z[:, i, :, :current_patch_len]
            patch_counts[:, :, start:end] += 1

        output /= patch_counts
        output = torch.reshape(output, (output.shape[0], output.shape[2], output.shape[1]))
        return output   # output: [bs, c_out, context_window]

    def forward(self, z, bases):  
        """
        [修改] 核心前向传播
        z: [Batch_Size * Patch_Num, Dim]  (经过投影后的输入特征)
        bases: [Num_Experts, Dim]         (来自 GNN 的专家基/聚类中心)
        """
        # 1. 计算 Logits (投影/点积)
        # z: [B, D], bases: [N, D] -> logits: [B, N]
        logits = torch.matmul(z, bases.T)
        
        # 2. 计算能量/亲和度 (对应 TFPS 论文的 Subspace Energy)
        # s = (z * D)^2
        s = torch.pow(logits, 2)
        
        # 3. 获取当前空间的维度 (用于 eta 缩放)
        # 使用传入 bases 的维度，而不是 init 中的 self.d，更加稳健
        current_d = bases.shape[1]
        
        # 4. 归一化处理 (TFPS 逻辑)
        # s 加上平滑项 eta
        s = (s + self.eta * current_d) / ((self.eta + 1) * current_d)
        
        # 转换为概率分布 (Row Normalization)
        s = (s.t() / torch.sum(s, 1)).t()  # [Batch, Num_Experts]
        
        return s, z

    def total_loss(self, pred, target, beta): 
        """
        [修改] 计算聚类损失
        pred: 预测的概率分布 (Q)
        target: 目标分布 (P, 通常由 target_distribution 函数计算)
        beta: 权重系数
        """
        # [移除] D_constraint1 和 D_constraint2
        # 正交性约束现在由 SparseMoE.py 中的 OrthoLoss 负责
        
        # Subspace clustering loss (KL Divergence)
        kl_loss = F.kl_div(pred.log(), target, reduction='batchmean')

        # Total_loss
        total_loss = beta * kl_loss

        return total_loss