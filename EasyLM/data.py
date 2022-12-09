import dataclasses
import pprint
from functools import partial

from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from transformers import AutoTokenizer
from datasets import load_dataset


class HuggingFacePretrainedTokenizer(object):
    """ Wrapper around HuggingFace's pretrained tokenizer. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.name = 'EleutherAI/gpt-j-6B'
        config.bos_token = '<|endoftext|>'
        config.eos_token = '<|endoftext|>'
        config.unk_token = '<|unknown|>'
        config.sep_token = '<|sep|>'
        config.pad_token = '<|pad|>'
        config.cls_token = '<|cls|>'
        config.mask_token = '<|mask|>'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            bos_token=self.config.bos_token,
            eos_token=self.config.eos_token,
            sep_token=self.config.sep_token,
            unk_token=self.config.unk_token,
            pad_token=self.config.pad_token,
            cls_token=self.config.cls_token,
            mask_token=self.config.mask_token,
        )

    @property
    def tokenizer(self):
        return self._tokenizer


class C4Dataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.tokenizer = HuggingFacePretrainedTokenizer.get_default_config()
        config.seq_length = 1024
        config.split = 'train'
        config.batch_size = 8
        config.vocab_pad_size = 256

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self._hf_tokenizer = HuggingFacePretrainedTokenizer(self.config.tokenizer)
        self._dataset = load_dataset('c4', 'en', split=self.config.split, streaming=True)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        while True:
            tokens = []
            for example in self._dataset:
                tokens.extend(self.tokenizer.encode(example['text']))
                tokens.append(self.tokenizer.eos_token_id)
                while len(tokens) > chunk_size:
                    yield {
                        'tokens': np.array(tokens[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        )
                    }
                    tokens = tokens[chunk_size:]

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._hf_tokenizer.tokenizer

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        """ Pad the vocab size so it can be evenly partitioned. """
        return (
            int(np.ceil(len(self.tokenizer) / self.config.vocab_pad_size))
            * self.config.vocab_pad_size
        )
