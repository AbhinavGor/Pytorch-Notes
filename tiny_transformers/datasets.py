class GenerateDataset:
    def __init__(self, encoded_text, block_size):
        self.encoded_text = encoded_text
        self.block_size = block_size

    def __len__(self):
        return len(self.encoded_text) - self.block_size - 1

    def __getitem__(self, idx):
        # x: [block_size], y: same length, shifted by 1 in the original stream
        x = self.encoded_text[idx : idx + self.block_size]
        y = self.encoded_text[idx + 1 : idx + 1 + self.block_size]
        return x, y
