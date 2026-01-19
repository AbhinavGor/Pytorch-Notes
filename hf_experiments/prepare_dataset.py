from collections import defaultdict
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(path, name=None, save_path="data/wikitext2_distilgpt2_block128", block_size=128):
    dataset = get_dataset(path, name)
    tokenized_dataset = tokenize_dataset(dataset=dataset)
    
    def group_texts(examples):
        # Concatenate over the batch for *all* fields (input_ids, attention_mask, ...)
        
        concatenated = {}
        for k, v in examples.items():
            # v is a list of lists; flatten it
            flat_list = []
            for seq in v:
                flat_list.extend(seq)
            concatenated[k] = flat_list
        
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated.items()
        }
        return result
    
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )
    
    lm_dataset.save_to_disk(save_path)
    
    return lm_dataset

def get_dataset(path, name=None):
    if name is None:
        return load_dataset(path)
    return load_dataset(path, name)

def tokenize_dataset(model_name='distilgpt2', dataset=get_dataset("wikitext", "wikitext-2-raw-v1")):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
    
    return dataset

if __name__ == "__main__":
    prepare_dataset("wikitext", "wikitext-2-raw-v1", save_path="data/wikitext2_distilgpt2_block128", block_size=128)