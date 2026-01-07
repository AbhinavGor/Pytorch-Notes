import torch
from attention import SelfAttention, MultiHeadSelfAttention

def test_self_attention_shape():
    B, T, D = 128, 1024, 1024
    
    model = SelfAttention(T)

    x = torch.randn(B, T, D)

    attention, _ = model(x)

    assert attention.shape == x.shape

def test_row_wise_sums():
    B, T, D = 128, 1024, 1024
    
    model = SelfAttention(T)

    x = torch.randn(B, T, D)

    attention, weights = model(x)
    
    row_sums = weights.sum(dim=-1)
    ones = torch.ones_like(row_sums)

    assert torch.allclose(row_sums, ones, atol=1e-5)

def test_multi_head_self_attention():
    B, T, D = 128, 1024, 1024
    h = 8
    
    model = MultiHeadSelfAttention(T, h)
    
    x = torch.randn(B, T, D)
    
    output = model(x)
    assert x.shape == output.shape
    
if __name__ == "__main__":
    test_row_wise_sums()