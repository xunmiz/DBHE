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
        # self.eta = eta # [修改] 移除 eta，不再使用固定的平滑参数
        self.c_out = c_out
        self.bs = bs
        self.patch_len = patch_len
        self.stride = stride

        # [新增] 可学习的温度系数 (Learnable Temperature)
        # 初始值设为 0.5 (或者更小，如 0.1，视数据敏感度而定)
        # 它可以自动调节聚类分布的尖锐程度 (Sharpness)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

        # [保留] 辅助元数据
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
        [修改] 基于余弦相似度和温度系数的前向传播
        z: [Batch_Size * Patch_Num, Dim]  (经过投影后的输入特征)
        bases: [Num_Experts, Dim]         (来自 Hypernetwork 的 Routing Embeddings)
        """
        # [改进 1] L2 归一化 (L2 Normalize)
        # 归一化后，点积即为余弦相似度，忽略幅度影响，只关注波形方向
        z_norm = F.normalize(z, p=2, dim=1)
        bases_norm = F.normalize(bases, p=2, dim=1)
        
        # [改进 2] 计算余弦相似度 (Cosine Similarity)
        # logits 范围在 [-1, 1] 之间
        # z_norm: [B, D], bases_norm: [N, D] -> logits: [B, N]
        logits = torch.matmul(z_norm, bases_norm.T)
        
        # [改进 3] 温度缩放 (Temperature Scaling)
        # 限制温度最小值为 1e-3 防止除零溢出
        # 类似于 SimCLR 或 CLIP 中的逻辑：logits / temp
        temp = torch.clamp(self.temperature, min=1e-3)
        scaled_logits = logits / temp
        
        # [改进 4] Softmax 概率分布
        # 替代原先的 power(2) 和行归一化
        s = F.softmax(scaled_logits, dim=-1)  # [Batch, Num_Experts]
        
        return s, z

    def total_loss(self, pred, target, beta): 
        """
        计算聚类损失
        pred: 预测的概率分布 (Q)
        target: 目标分布 (P, 通常由 target_distribution 函数计算)
        beta: 权重系数
        """
        # Subspace clustering loss (KL Divergence)
        kl_loss = F.kl_div(pred.log(), target, reduction='batchmean')

        # Total_loss
        total_loss = beta * kl_loss

        return total_loss