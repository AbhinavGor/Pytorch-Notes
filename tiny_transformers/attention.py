import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)  # W^Q
        self.k = nn.Linear(d_model, d_model, bias=False)  # W^K
        self.v = nn.Linear(d_model, d_model, bias=False)  # W^V

    def forward(self, x, causal=False):  # x: [B, T, D]
        B, T, D = x.shape
        Q = self.q(x)  # [B, T, D]
        K = self.k(x)  # [B, T, D]
        V = self.v(x)  # [B, T, D]
        
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(d_k)   # [B, T, T]
        
        if causal:
            # mask out positions j > i
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)
        
        return attention, weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, h, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.d_head = d_model//h
        self.h = h
        
        self.attention = nn.ModuleList([SelfAttention(d_model//h) for _ in range(h)])
        self.output = nn.Linear(self.d_head*h, self.d_head*h, bias=False)
        
    def forward(self, x, causal=False):
        
        B, T, D = x.shape
        d_head = D // self.h
        x = x.view(B, T, self.h, d_head)      # [B, T, h, d_head]
        x = x.transpose(1, 2)            # [B, h, T, d_head]

        attention_h = []
        
        for head_idx in range(self.h):
            x_i = x[:, head_idx, :, :]
            out_i, _ = self.attention[head_idx](x_i, causal)
            attention_h.append(out_i)
        
        attention = torch.cat(attention_h, dim=-1)

        return self.output(attention)
        
if __name__ == "__main__":
    x = torch.randn(128, 512, 512)

    model = SelfAttention(512)

    print(model(x))