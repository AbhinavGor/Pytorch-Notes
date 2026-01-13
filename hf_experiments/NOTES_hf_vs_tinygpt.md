## TinyGPT vs distilgpt2 (HF)

### High-level differences

* **Scale**: distilgpt2 is a much larger model (more layers, more heads, larger embedding size), so it has far more capacity than my TinyGPT.
* **Tokenization**: TinyGPT is **character-level**; distilgpt2 uses a **subword/BPE tokenizer** with a much larger vocab.
* **Context**: TinyGPT was trained with a small context window (`block_size`), while distilgpt2 supports sequences up to 1024 tokens (`n_ctx` / `n_positions`).
* **Config vs training hyperparams**: In TinyGPT, my `config` mixed **architecture** and **training** settings (e.g., `batch_size`, `block_size`). In HF, `model.config` only stores **architecture/behavior** (layers, heads, hidden size, vocab, max positions). Training hyperparams (batch size, training block size, LR, etc.) live in the training script / `TrainingArguments`, not in `model.config`.

### Concrete config comparisons

| Aspect               | TinyGPT                                     | distilgpt2 (HF)                                                 |
| -------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| Tokenization         | Raw characters                              | BPE / subword tokens via `AutoTokenizer`                        |
| Vocab size           | ≈ 65 (printable chars, newline, etc.)       | > 50k tokens (`config.vocab_size`)                              |
| Context length       | `block_size ≈ 128` (training + max context) | `n_ctx` / `n_positions = 1024` (architectural max context)      |
| # Transformer layers | `n_layer = 4`                               | `n_layer = 6`                                                   |
| # Attention heads    | `n_head = 4`                                | `n_head = 12`                                                   |
| d_model              | `d_model = [my TinyGPT value]`              | `n_embd` (hidden size) = 768                                    |
| Head dimension       | `d_model / n_head` (smaller per-head size)  | `768 / 12 = 64` per head                                        |
| Config contents      | Mixed model + training hyperparams          | Only model architecture/behavior (no `batch_size`, no LR, etc.) |

Mentally, I can view distilgpt2 as a “scaled-up, BPE-token, properly pretrained” version of the same Transformer ideas I used in TinyGPT: same basic blocks (self-attention + MLP + layernorm), just with larger dimensions, longer context, and a much richer tokenizer.
