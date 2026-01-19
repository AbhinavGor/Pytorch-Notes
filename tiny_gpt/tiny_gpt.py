import torch
import torch.nn as nn

from tiny_transformers.blocks import TransformerBlock
from tiny_gpt.model_config import ModelConfig

class CharTokenizer():
    def __init__(self, text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            self.text = f.read()

        # get sorted list of unique characters
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)

        # char ↔ id mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        """string -> list of int ids"""
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        """list of int ids -> string"""
        return "".join(self.itos[i] for i in ids)

class TinyGPT(nn.Module):
    def __init__(self, d_model, n_layers, h, vocab_size, block_size):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.h = h
        
        self.token_em = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_em = nn.Embedding(self.block_size, self.d_model)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.d_model, self.h) for _ in range(n_layers)])
        
        self.projection = nn.Linear(self.d_model, self.vocab_size)
        
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.n_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.h = config.n_head
        
        self.token_em = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_em = nn.Embedding(self.block_size, self.d_model)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.d_model, self.h) for _ in range(self.n_layers)])
        
        self.projection = nn.Linear(self.d_model, self.vocab_size)
            
    def forward(self, x, causal=True):
        # x: [B, T] token ids
        B, T = x.shape
        assert T <= self.block_size

        # token + positional embeddings
        token_ems = self.token_em(x)                 # [B, T, d_model]
        positions = torch.arange(T, device=x.device) # [T]
        pos_ems = self.pos_em(positions)             # [T, d_model]
        x = token_ems + pos_ems.unsqueeze(0)         # [B, T, d_model]

        # transformer blocks
        for block in self.transformer_blocks:
            x = block(x, causal=causal)

        # final projection to vocab
        return self.projection(x)                    # [B, T, vocab_size]

def get_batch(split, train_data, val_data, config: ModelConfig):
    # split: "train" or "val"
    source = train_data if split == "train" else val_data
    T = source.size(0)
    ix = torch.randint(0, T - config.block_size - 1, (config.batch_size,))
    x = torch.stack([source[i : i + config.block_size] for i in ix])          # [B, T]
    y = torch.stack([source[i + 1 : i + 1 + config.block_size] for i in ix])  # [B, T]
    return x, y

def evaluate(model, split="val", steps=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(split)        # [B, T], [B, T]
            logits = model(x)              # [B, T, vocab_size]
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                y.view(B * T),
            )
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    print(f"{split} loss: {avg_loss:.4f}")
    model.train()
    return avg_loss
 
def generate(model, tokenizer, start_text, max_new_tokens=200):
    model.eval()
    device = next(model.parameters()).device

    # encode prompt
    idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -model.block_size:]

            # forward
            logits = model(idx_cond)          # [1, T, vocab_size]
            logits = logits[:, -1, :]         # last time step: [1, vocab_size]

            # sample next token
            probs = torch.softmax(logits, dim=-1)  # [1, vocab_size]
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # append
            idx = torch.cat([idx, next_id], dim=1)

    model.train()
    return tokenizer.decode(idx[0].tolist())
       
if __name__ == "__main__":
    # 1. Build tokenizer and encode the whole corpus
    ct = CharTokenizer("data/tiny_shakespeare.txt")
    data = torch.tensor(ct.encode(ct.text), dtype=torch.long)  # [T]
    vocab_size = ct.vocab_size

    # 2. Train/val split
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data   = data[n:]

    block_size = 128
    batch_size = 64

    # 3. Create model
    d_model   = 128
    n_layers  = 4
    n_heads   = 4

    model = TinyGPT(
        d_model=d_model,
        n_layers=n_layers,
        h=n_heads,
        vocab_size=vocab_size,
        block_size=block_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # 4. Training loop (very simple)
    for step in range(10_000):
        x, y = get_batch("train")                # x, y: [B, T]
        logits = model(x, causal=True)                        # [B, T, vocab_size]

        B, T, V = logits.shape
        loss = criterion(
            logits.view(B * T, V),
            y.view(B * T),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f}")
            
    evaluate(model, "val", steps=500)

    prompt = "ROMEO:"
    sample = generate(model, ct, prompt, max_new_tokens=300)
    print(sample)
    

            