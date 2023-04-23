import dataclasses
import pprint
import time
from functools import partial
import json

import h5py
import mlxu
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from datasets import load_dataset


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_eos_token = True
        config.prepend_text = ''
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example):
        token_buffer = []
        loss_mask_buffer = []
        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    yield {
                        'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }, metrics
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
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.start_seek_loc = 0
        config.h5_field = ''
        config.h5_start_index = 0
        config.seq_length = 1024
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        if self.config.h5_field != '':
            h5_file = h5py.File(
                mlxu.open_file(self.config.path, 'rb', cache_type='block'), 'r'
            )
            length = h5_file[self.config.h5_field].shape[0]
            index = self.config.h5_start_index % length
            while True:
                data = self.parse_json(
                    mlxu.array_to_text(
                        h5_file[self.config.h5_field][index]
                    )
                )
                if data is not None:
                    yield 0, index, data

                index += 1
                if index >= length:
                    index = 0
        else:
            with mlxu.open_file(self.config.path, 'r') as fin:
                fin.seek(self.config.start_seek_loc)
                index = 0
                while True:
                    line = fin.readline()
                    loc = fin.tell()
                    if not line:   # Reached EOF
                        index = 0
                        fin.seek(0)
                        continue

                    data = self.parse_json(line)
                    if data is not None:
                        # JSON parsing succeeded
                        yield loc, index, data
                    index += 1

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        total_tokens = 0
        last_time = 0.0
        for loc, index, example in self.json_iterator():
            tokens, loss_masks = self.text_processor(example)
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size:
                total_tokens += chunk_size
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': total_tokens,
                    'dataset_throughput_tps': chunk_size / (time.time() - last_time),
                }
                last_time = time.time()
                yield {
                    'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }, metrics
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
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)
