import torch
import torch.nn as nn
from transformers.attention import MultiHeadSelfAttention

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        
        self.linear1 = nn.Linear(self.d_model, 4*self.d_model)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(4*self.d_model, self.d_model)
        
        
    def forward(self, x):
        return self.linear2(self.activation1(self.linear1(x)))
        
class TransformerBlock(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h = h
        self.d_model = d_model
        
        self.mha = MultiHeadSelfAttention(self.d_model, self.h)
        self.mlp = MLP(self.d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, causal=True):
        attention = self.mha(x, causal=causal)
        
        norm1 = self.ln1(x+attention)
        
        ff = self.mlp(norm1)
        
        norm2 = self.ln2(norm1+ff)
        
        return norm2
    
    
if __name__ == "__main__":
    x = torch.randn(128, 1024, 1024)
    
    transformer = TransformerBlock(1024, 8)
    
    print(transformer(x))
    