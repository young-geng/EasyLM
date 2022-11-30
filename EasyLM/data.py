import dataclasses
import pprint
from functools import partial

from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from transformers import AutoTokenizer
from datasets import load_dataset


class C4Dataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.tokenizer = 'EleutherAI/gpt-j-6B'
        config.seq_length = 512
        config.split = 'train'
        config.batch_size = 8
        config.vocab_pad_size = 64

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self._dataset = load_dataset('c4', 'en', split=self.config.split, streaming=True)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        while True:
            tokens = []
            for example in self._dataset:
                tokens.extend(self._tokenizer.encode(example['text']))
                tokens.append(self._tokenizer.eos_token_id)
                while len(tokens) > chunk_size:
                    yield {
                        'tokens': np.array(tokens[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        )
                    }
                    tokens = tokens[chunk_size:]

    @property
    def vocab_size(self):
        """ Reserve some extra tokens for downstream tasks. """
        return (
            int(np.ceil(self._tokenizer.vocab_size / self.config.vocab_pad_size))
            * self.config.vocab_pad_size
        )

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def dataset(self):
        return self._dataset
