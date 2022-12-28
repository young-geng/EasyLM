import dataclasses
import pprint
from functools import partial
import re
import os
from threading import Lock

import absl.logging
from tqdm import tqdm, trange
import numpy as np
import wandb
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from flask import Flask, request


class LMServer(object):
    """ HTTP server for serving langauge models. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.name = 'lm_server'
        config.host = '0.0.0.0'
        config.port = 5007
        config.batch_size = 1
        config.logging = True
        config.pre_compile = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self.lock = Lock()
        self.app = Flask(self.config.name)
        self.app.post('/loglikelihood')(self.loglikelihood)
        self.app.post('/loglikelihood-rolling')(self.loglikelihood_rolling)
        self.app.post('/generate')(self.generate)

    @staticmethod
    def loglikelihood_fn(prefix_text, text):
        raise NotImplementedError()

    @staticmethod
    def loglikelihood_rolling_fn(text):
        raise NotImplementedError()

    @staticmethod
    def generate_fn(text, temperature):
        raise NotImplementedError()

    @staticmethod
    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    def loglikelihood(self):
        with self.lock:
            data = request.get_json()
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Log Likelihood Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )

            text = data['text']
            if 'prefix_text' not in data:
                prefix_text = ['' for _ in text]
            else:
                prefix_text = data['prefix_text']

            log_likelihood = []
            is_greedy = []
            for i in range(0, len(text), self.config.batch_size):
                batch_prefix_text = prefix_text[i:i + self.config.batch_size]
                batch_text = text[i:i + self.config.batch_size]
                batch_size = len(batch_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_prefix_text.extend(['a' for _ in range(extra)])
                    batch_text.extend(['a' for _ in range(extra)])

                batch_log_likelihood, batch_is_greedy = self.loglikelihood_fn(
                    batch_prefix_text, batch_text
                )
                batch_log_likelihood = self.to_list(batch_log_likelihood)
                batch_is_greedy = self.to_list(batch_is_greedy)
                log_likelihood.extend(batch_log_likelihood[:batch_size])
                is_greedy.extend(batch_is_greedy[:batch_size])

            output = {
                'prefix_text': prefix_text,
                'text': text,
                'log_likelihood': log_likelihood,
                'is_greedy': is_greedy,
            }
            if self.config.logging:
                absl.logging.info(
                '\n========= Output ========= \n'
                + pprint.pformat(output) + '\n'
            )

        return output

    def loglikelihood_rolling(self):
        with self.lock:
            data = request.get_json()
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Log Likelihood Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )

            text = data['text']
            log_likelihood = []
            is_greedy = []
            for i in range(0, len(text), self.config.batch_size):
                batch_text = text[i:i + self.config.batch_size]
                batch_size = len(batch_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_text.extend(['a' for _ in range(extra)])

                batch_log_likelihood, batch_is_greedy = self.loglikelihood_rolling_fn(
                    batch_text
                )
                batch_log_likelihood = self.to_list(batch_log_likelihood)
                batch_is_greedy = self.to_list(batch_is_greedy)
                log_likelihood.extend(batch_log_likelihood[:batch_size])
                is_greedy.extend(batch_is_greedy[:batch_size])

            output = {
                'text': text,
                'log_likelihood': log_likelihood,
                'is_greedy': is_greedy,
            }
            if self.config.logging:
                absl.logging.info(
                '\n========= Output ========= \n'
                + pprint.pformat(output) + '\n'
            )

        return output

    def generate(self):
        with self.lock:
            data = request.get_json()
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Generate Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )
            prefix_text = data['prefix_text']
            temperature = data.get('temperature', 1.0)

            output_text = []
            for i in range(0, len(prefix_text), self.config.batch_size):
                batch_prefix_text = prefix_text[i:i + self.config.batch_size]
                batch_size = len(batch_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_prefix_text.extend(['a' for _ in range(extra)])

                batch_output_text = self.generate_fn(batch_prefix_text, temperature)
                output_text.extend(self.to_list(batch_output_text))

            output = {
                'prefix_text': prefix_text,
                'temperature': temperature,
                'output_text': output_text,
            }
            if self.config.logging:
                absl.logging.info(
                    '\n========= Output ========= \n'
                    + pprint.pformat(output) + '\n'
                )
        return output

    def run(self):
        if self.config.pre_compile:
            pre_compile_data = ['a' for _ in range(self.config.batch_size)]
            self.loglikelihood_fn(pre_compile_data, pre_compile_data)
            self.generate_fn(pre_compile_data, 1.0)
            self.loglikelihood_rolling_fn(pre_compile_data)
        self.app.run(host=self.config.host, port=self.config.port)

