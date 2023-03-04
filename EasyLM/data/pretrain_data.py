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
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.seq_length = 1024
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.field = 'text'
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
        fields = self.config.field.split(',')
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for example in self._dataset:
                for field in fields:
                    if field.startswith('[') and field.endswith(']'):
                        # No loss for this field.
                        field = field[1:-1]
                        mask = 0.0
                    else:
                        mask = 1.0
                    tokens = self.tokenizer.encode(example[field])
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])

                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(1.0)
                while len(token_buffer) > chunk_size:
                    yield {
                        'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

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
    """ HDF5 dataset, where text is encoded as uint8 arrays. We use
        mlxu.array_to_text to convert the uint8 array back to text. Although
        this is not the most straightforward way to store text, hdf5 does
        support indexing, which makes it easier to load data at particular
        index.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.field = 'text'
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
            # Loading from GCS. Using block cache is important here because
            # hdf5 does not read the file sequentially.
            h5_file = h5py.File(
                mlxu.open_file(self.config.path, 'rb', cache_type='block'), 'r'
            )
        else:
            h5_file = h5py.File(self.config.path, 'r')

        chunk_size = self.config.batch_size * self.config.seq_length
        fields = self.config.field.split(',')
        token_buffer = []
        loss_mask_buffer = []
        while True:
            for field in fields:
                if field.startswith('[') and field.endswith(']'):
                    # No loss for this field.
                    field = field[1:-1]
                    mask = 0.0
                else:
                    mask = 1.0
                tokens = self.tokenizer.encode(
                    mlxu.array_to_text(h5_file[field][self.index])
                )
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)
            while len(token_buffer) > chunk_size:
                yield {
                    'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

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
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.fields_from_example = ''
        config.field = 'text'
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
        token_buffer = []
        loss_mask_buffer = []
        for example in self.json_iterator():
            if self.config.fields_from_example != '':
                fields = example[self.config.fields_from_example].split(',')
            else:
                fields = self.config.field.split(',')
            for field in fields:
                if field.startswith('[') and field.endswith(']'):
                    # No loss for this field.
                    field = field[1:-1]
                    mask = 0.0
                else:
                    mask = 1.0
                tokens = self.tokenizer.encode(example[field])
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)
            while len(token_buffer) > chunk_size:
                yield {
                    'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

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
