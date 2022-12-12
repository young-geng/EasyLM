import dataclasses
import pprint
from functools import partial
from io import BytesIO

import gcsfs
import h5py
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

        config.vocab_pad_size = 256

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

    @property
    def vocab_size(self):
        """ Pad the vocab size so it can be evenly partitioned. """
        return (
            int(np.ceil(len(self.tokenizer) / self.config.vocab_pad_size))
            * self.config.vocab_pad_size
        )


class HuggingfaceDataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.tokenizer = HuggingFacePretrainedTokenizer.get_default_config()
        config.seq_length = 1024
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.field = 'text'
        config.streaming=True
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None

        self._hf_tokenizer = HuggingFacePretrainedTokenizer(self.config.tokenizer)
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

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
        return self._hf_tokenizer.vocab_size


class H5Dataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.field = 'text'
        config.tokenizer = HuggingFacePretrainedTokenizer.get_default_config()
        config.seq_length = 1024
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_index=0):
        self.config = self.get_default_config(config)
        assert self.config.path != ''

        if self.config.path.startswith('gs://'):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type='block'),
                'r'
            )
        else:
            self.h5_file = h5py.File(self.config.path, 'r')

        self._hf_tokenizer = HuggingFacePretrainedTokenizer(self.config.tokenizer)
        self.index = 0

    def __getstate__(self):
        return self.config, self.index

    def __setstate__(self, state):
        config, start_index = state
        self.__init__(config, start_index)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        tokens = []
        while True:
            with BytesIO(self.h5_file[self.config.field][self.index]) as fin:
                text = fin.read().decode('utf-8')

            self.index = (self.index + 1) % self.h5_file[self.config.field].shape[0]
            tokens.extend(self.tokenizer.encode(text))
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
        return self._hf_tokenizer.vocab_size


class PretrainDataset(object):
    """ Pretraining datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.dataset_type = 'huggingface'
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.h5_dataset = H5Dataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config):
        config = cls.get_default_config(config)
        if config.dataset_type == 'huggingface':
            return HuggingfaceDataset(config.huggingface_dataset)
        elif config.dataset_type == 'h5':
            return H5Dataset(config.h5_dataset)
        else:
            raise ValueError(f'Unknown dataset type: {config.dataset_type}')

    def __init__(self):
        raise ValueError('PretrainDataset is a static class and should not be instantiated.')
