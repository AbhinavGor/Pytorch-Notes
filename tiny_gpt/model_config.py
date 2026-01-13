class ModelConfig:
    def __init__(self, config):
        self._vocab_size = config["vocab_size"]
        self._d_model    = config["d_model"]
        self._n_layer    = config["n_layer"]
        self._n_head     = config["n_head"]
        self._block_size = config["block_size"]
        self._batch_size = config["batch_size"]

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def d_model(self):
        return self._d_model

    @property
    def n_layer(self):
        return self._n_layer

    @property
    def n_head(self):
        return self._n_head

    @property
    def block_size(self):
        return self._block_size
    
    @property
    def batch_size(self):
        return self._batch_size