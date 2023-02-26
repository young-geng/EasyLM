import dataclasses
import pprint
from functools import partial
import json

import mlxu
import h5py
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from datasets import load_dataset


class PretrainDataset(object):
    """ Pretraining datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.h5_dataset = H5Dataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        if config.type == 'huggingface':
            return HuggingfaceDataset(config.huggingface_dataset, tokenizer, **kwargs)
        elif config.type == 'h5':
            return H5Dataset(config.h5_dataset, tokenizer, **kwargs)
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('PretrainDataset is a static class and should not be instantiated.')


class HuggingfaceDataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.seq_length = 1024
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.field = 'text'
        config.field_sep = ' '
        config.streaming = True
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None

        self._tokenizer = tokenizer
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        while True:
            tokens = []
            for example in self._dataset:
                text = self.config.field_sep.join(
                    [example[key] for key in self.config.field.split(',')]
                )
                tokens.extend(self.tokenizer.encode(text))
                tokens.append(self.tokenizer.eos_token_id)
                while len(tokens) > chunk_size:
                    yield {
                        'tokens': np.array(tokens[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        )
                    }
                    tokens = tokens[chunk_size:]

    def __getstate__(self):
        return self.config, self.tokenizer

    def __setstate__(self, state):
        config, tokenizer = state
        self.__init__(config, tokenizer)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class H5Dataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.field = 'text'
        config.field_sep = ' '
        config.seq_length = 1024
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, start_index=0):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self.index = start_index

    def __iter__(self):
        if self.config.path.startswith('gs://'):
            # Loading from GCS
            h5_file = h5py.File(mlxu.open_file(self.config.path, 'rb'), 'r')
        else:
            h5_file = h5py.File(self.config.path, 'r')

        chunk_size = self.config.batch_size * self.config.seq_length
        tokens = []
        fields = self.config.field.split(',')
        while True:
            text = self.config.field_sep.join(
                [mlxu.array_to_text(h5_file[field][self.index]) for field in fields]
            )
            self.index = (self.index + 1) % h5_file[fields[0]].shape[0]
            tokens.extend(self.tokenizer.encode(text))
            tokens.append(self.tokenizer.eos_token_id)
            while len(tokens) > chunk_size:
                yield {
                    'tokens': np.array(tokens[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    )
                }
                tokens = tokens[chunk_size:]

    def __getstate__(self):
        return self.config, self.tokenizer, self.index

    def __setstate__(self, state):
        config, tokenizer, start_index = state
        self.__init__(config, tokenizer, start_index)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class JsonDataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.field = 'text'
        config.field_sep = ' '
        config.seq_length = 1024
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer

    def json_iterator(self):
        while True:
            with mlxu.open_file(self.config.path, 'r') as fin:
                for line in fin:
                    if not line or line == '\n':
                        continue
                    try:
                        data = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        print(f'Error parsing json line:\n{line}')
                        continue
                    yield data

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        tokens = []
        fields = self.config.field.split(',')
        for example in self.json_iterator():
            text = self.config.field_sep.join([example[field] for field in fields])
            tokens.extend(self.tokenizer.encode(text))
            tokens.append(self.tokenizer.eos_token_id)
            while len(tokens) > chunk_size:
                yield {
                    'tokens': np.array(tokens[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    )
                }
                tokens = tokens[chunk_size:]

    def __getstate__(self):
        return self.config, self.tokenizer

    def __setstate__(self, state):
        config, tokenizer = state
        self.__init__(config, tokenizer)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return len(self.tokenizer)
