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
        config.tokenizer = 'bert-base-uncased'
        config.seq_length = 512
        config.split = 'train'
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.dataset = load_dataset('c4', 'en', split=self.config.split, streaming=True)


    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        tokens = []
        for example in self.dataset:
            tokens.extend(self.tokenizer.encode(example['text'] + '[eod]'))
            if len(tokens) > chunk_size:
                yield {
                    'tokens': np.array(tokens[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    )
                }
                tokens = tokens[chunk_size:]
