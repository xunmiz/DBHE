import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# ==========================================
# 1. Functional Expert 
# ==========================================
def functional_expert(x, w1, b1, w2, b2, training=True, dropout_p=0.3):
    is_batched_params = w1.dim() == 3

    if is_batched_params:
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, D_in]
        
        # Linear: x @ W.T + b
        x = torch.bmm(x, w1.transpose(1, 2)) + b1.unsqueeze(1)
    else:
       
        x = F.linear(x, w1, b1)

    x = F.relu(x)

    # --- Layer 2 ---
    if is_batched_params:
        x = torch.bmm(x, w2.transpose(1, 2)) + b2.unsqueeze(1)
        if x.size(1) == 1:
            x = x.squeeze(1) 
    else:
        x = F.linear(x, w2, b2)

    x = F.dropout(x, p=dropout_p, training=training)
    return x


class HyperNetwork(nn.Module):
    def __init__(self, num_experts, data_n_embd, d_embed=256, rank=64):
        super().__init__()
        self.data_n_embd = data_n_embd
        self.hidden_dim = 4 * data_n_embd
        self.rank = rank
        self.num_experts = num_experts

   
        self.expert_embeddings = nn.Parameter(torch.empty(num_experts, d_embed))
        init.normal_(self.expert_embeddings, mean=0, std=0.02)
        
        

       
        self.routing_proj = nn.Sequential(
            nn.Linear(d_embed, d_embed * 4),  
            nn.ReLU(),                                    
            nn.Linear(d_embed * 4, d_embed)   
        )

        
        nn.init.kaiming_normal_(self.routing_proj[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.routing_proj[0].bias)
        nn.init.zeros_(self.routing_proj[2].weight)
        nn.init.zeros_(self.routing_proj[2].bias)

     
        self.shared_w1 = nn.Parameter(torch.empty(self.hidden_dim, data_n_embd))
        self.shared_b1 = nn.Parameter(torch.zeros(self.hidden_dim))
        self.shared_w2 = nn.Parameter(torch.empty(data_n_embd, self.hidden_dim))
        self.shared_b2 = nn.Parameter(torch.zeros(data_n_embd))
        
        init.kaiming_uniform_(self.shared_w1, a=5**0.5)
        init.kaiming_uniform_(self.shared_w2, a=5**0.5)

     
        self.gen_w1_A = nn.Linear(d_embed, self.hidden_dim * rank)
        self.gen_w1_B = nn.Linear(d_embed, rank * data_n_embd)
        
        self.gen_w2_A = nn.Linear(d_embed, data_n_embd * rank)
        self.gen_w2_B = nn.Linear(d_embed, rank * self.hidden_dim)
        
        self.gen_b1 = nn.Linear(d_embed, self.hidden_dim)
        self.gen_b2 = nn.Linear(d_embed, data_n_embd)

        with torch.no_grad():
            for layer in [self.gen_w1_A, self.gen_w1_B, self.gen_w2_A, self.gen_w2_B]:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_embeddings(self):
       
        return self.expert_embeddings

    def get_routing_bases(self):
       
        raw_emb = self.get_embeddings()
        return self.routing_proj(raw_emb)

    def get_ortho_loss(self):
        
        bases = self.get_routing_bases()
       
        norms = torch.norm(bases, p=2, dim=1, keepdim=True)
        normalized_bases = bases / (norms + 1e-8)
       
        sim_matrix = torch.matmul(normalized_bases, normalized_bases.T)
   
        identity = torch.eye(self.num_experts, device=bases.device)
  
        return torch.norm(sim_matrix - identity, p='fro') ** 2

    def generate_weights_from_embedding(self, emb):
      
        w1_a = self.gen_w1_A(emb).view(*emb.shape[:-1], self.hidden_dim, self.rank)
        w1_b = self.gen_w1_B(emb).view(*emb.shape[:-1], self.rank, self.data_n_embd)
        delta_w1 = torch.matmul(w1_a, w1_b)
        
        w2_a = self.gen_w2_A(emb).view(*emb.shape[:-1], self.data_n_embd, self.rank)
        w2_b = self.gen_w2_B(emb).view(*emb.shape[:-1], self.rank, self.hidden_dim)
        delta_w2 = torch.matmul(w2_a, w2_b)
        
        delta_b1 = self.gen_b1(emb).view(*emb.shape[:-1], self.hidden_dim)
        delta_b2 = self.gen_b2(emb).view(*emb.shape[:-1], self.data_n_embd)

        return delta_w1, delta_b1, delta_w2, delta_b2

    def generate_weights(self, expert_idx, embeddings):
    
        emb = embeddings[expert_idx]
        return self.generate_weights_from_embedding(emb)


# ==========================================
# 3. Router (Cluster Logic) 
class NoisyTopkRouter_Cluster(nn.Module):
    def __init__(self, top_k):
        super(NoisyTopkRouter_Cluster, self).__init__()
        self.top_k = top_k

    def forward(self, logits):
        noise = torch.randn_like(logits) * F.softplus(logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


# ==========================================
# 4. SparseMoE (Main Module)
# ==========================================
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, d_embed, rank, k):
        super(SparseMoE, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.d_embed = d_embed
        self.rank = rank
        self.input_norm = nn.LayerNorm(n_embed)
        self.router = NoisyTopkRouter_Cluster(top_k)
        self.k = None
        
        
        self.hypernet = HyperNetwork(num_experts, n_embed, d_embed=self.d_embed, rank=self.rank)
      
        self.ortho_loss = torch.tensor(0.0)


    def forward(self, x, expert_logits):
        x_norm = self.input_norm(x)
        
        
        expert_embeddings = self.hypernet.get_embeddings()
        self.ortho_loss = self.hypernet.get_ortho_loss()

        gating_output, indices = self.router(expert_logits)
        
        final_output = torch.zeros_like(x)
        flat_x = x_norm.reshape(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

     
        for i in range(self.num_experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]

                w1, b1, w2, b2 = self.hypernet.generate_weights(i, expert_embeddings)

          
                expert_output = functional_expert(expert_input, w1, b1, w2, b2, training=self.training)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                flat_final_output = final_output.reshape(-1, final_output.size(-1))
                flat_final_output[expert_mask] += weighted_output.squeeze(1)
                final_output = flat_final_output.reshape(x.shape[0], x.shape[1], x.shape[2])

        return final_output + x

  
    def inference(self, x, expert_affinity):
        batch, seq, dim = x.shape
        x_norm = self.input_norm(x)
        k = self.k
      
        if expert_affinity.dim() == 3:
            alpha = expert_affinity.mean(dim=1) # [Batch, N]
        else:
            alpha = expert_affinity # [Batch, N]

        use_topk = (k is not None) and (k < self.num_experts) and (k > 0)

        if use_topk:
            topk_values, topk_indices = torch.topk(alpha, k=k, dim=-1)
            mask = torch.zeros_like(alpha).scatter_(-1, topk_indices, 1.0)
            alpha = alpha * mask
        
   
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)

        base_embeddings = self.hypernet.get_embeddings()
     
        mixed_embeddings = torch.matmul(alpha, base_embeddings)

        w1, b1, w2, b2 = self.hypernet.generate_weights_from_embedding(mixed_embeddings)

        out = functional_expert(x_norm, w1, b1, w2, b2, training=False)
        
        return out + x