from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd

from tiny_gpt.model_config import ModelConfig
from tiny_gpt.tiny_gpt import CharTokenizer, TinyGPT, evaluate, generate, get_batch

if __name__ == "__main__":    
    ct = CharTokenizer("data/tiny_shakespeare.txt")
    data = torch.tensor(ct.encode(ct.text), dtype=torch.long)  # [T]
    vocab_size = ct.vocab_size
    
    config = {
        'vocab_size': vocab_size,
        'd_model': 128,
        'n_layer': 4,
        'n_head': 4,
        'block_size': 128,
        'batch_size': 64,
        'log_loss': True
    }
    
    model_config = ModelConfig(config)
    model = TinyGPT(model_config)

    # 2. Train/val split
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    loss_vals = []
    
    # 4. Training loop (very simple)
    for step in range(10_000):
        x, y = get_batch("train", train_data, val_data, model_config)        # x, y: [B, T]
        logits = model(x, causal=True)                        # [B, T, vocab_size]

        B, T, V = logits.shape
        loss = criterion(
            logits.view(B * T, V),
            y.view(B * T),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if config['log_loss']:
            loss_vals.append([step, loss.item()])
            
        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f}")
            
    if config['log_loss']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"loss_log_{timestamp}.csv"
        loss_df = pd.DataFrame(loss_vals, columns=['step_number', 'loss_value'])
        loss_df.to_csv(filename, index=False)
        
    evaluate(model, "val", steps=500)
    
    prompt = "ROMEO:"
    sample = generate(model, ct, prompt, max_new_tokens=300)
    print(sample)