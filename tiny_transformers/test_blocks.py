import torch
from blocks import TransformerBlock

def test_transformer_block():
    x = torch.randn(128, 1024, 1024)
    
    transformer = TransformerBlock(1024, 8)
    
    out = transformer(x)
    assert out.shape == x.shape
    
    loss = out.sum()
    
    loss.backward()
    