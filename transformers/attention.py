import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)  # W^Q
        self.k = nn.Linear(d_model, d_model, bias=False)  # W^K
        self.v = nn.Linear(d_model, d_model, bias=False)  # W^V

    def forward(self, x):  # x: [B, T, D]
        Q = self.q(x)  # [B, T, D]
        K = self.k(x)  # [B, T, D]
        V = self.v(x)  # [B, T, D]
        
        d_k = len(K.size())
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(d_k)   # [B, T, T]
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)
        
        return attention, weights

if __name__ == "__main__":
    x = torch.randn(128, 512, 512)

    model = SelfAttention(512)

    print(model(x))