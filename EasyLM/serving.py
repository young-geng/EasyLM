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
        config.logging = False
        config.pre_compile = 'loglikelihood'
        config.default_temperature = 1.0
        config.default_max_length = 5000

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self.lock = Lock()
        self.app = Flask(self.config.name)
        self.app.post('/loglikelihood')(self.serve_loglikelihood)
        self.app.post('/loglikelihood-rolling')(self.serve_loglikelihood_rolling)
        self.app.post('/generate')(self.serve_generate)
        self.app.post('/greedy-until')(self.serve_greedy_until)
        self.app.route('/ready')(self.serve_ready)

    @staticmethod
    def loglikelihood(prefix_text, text):
        raise NotImplementedError()

    @staticmethod
    def loglikelihood_rolling(text):
        raise NotImplementedError()

    @staticmethod
    def generate(text, temperature):
        raise NotImplementedError()

    @staticmethod
    def greedy_until(prefix_text, until, max_length):
        raise NotImplementedError()

    @staticmethod
    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    def serve_loglikelihood(self):
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
            for i in trange(0, len(text), self.config.batch_size, ncols=0):
                batch_prefix_text = prefix_text[i:i + self.config.batch_size]
                batch_text = text[i:i + self.config.batch_size]
                batch_size = len(batch_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_prefix_text.extend(['a' for _ in range(extra)])
                    batch_text.extend(['a' for _ in range(extra)])

                batch_log_likelihood, batch_is_greedy = self.loglikelihood(
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

    def serve_loglikelihood_rolling(self):
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
            for i in trange(0, len(text), self.config.batch_size, ncols=0):
                batch_text = text[i:i + self.config.batch_size]
                batch_size = len(batch_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_text.extend(['a' for _ in range(extra)])

                batch_log_likelihood, batch_is_greedy = self.loglikelihood_rolling(
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

    def serve_generate(self):
        with self.lock:
            data = request.get_json()
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Generate Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )
            prefix_text = data['prefix_text']
            temperature = data.get('temperature', self.config.default_temperature)

            output_text = []
            for i in trange(0, len(prefix_text), self.config.batch_size, ncols=0):
                batch_prefix_text = prefix_text[i:i + self.config.batch_size]
                batch_size = len(batch_prefix_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_prefix_text.extend(['a' for _ in range(extra)])

                batch_output_text = self.generate(batch_prefix_text, temperature)
                output_text.extend(self.to_list(batch_output_text)[:batch_size])

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

    def serve_greedy_until(self):
        with self.lock:
            data = request.get_json()
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Greedy Until Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )
            prefix_text = data['prefix_text']
            until = data['until']
            max_length = data.get('max_length', self.config.default_max_length)

            output_text = []
            for i in range(0, len(prefix_text), self.config.batch_size):
                batch_prefix_text = prefix_text[i:i + self.config.batch_size]
                batch_until = until[i:i + self.config.batch_size]
                batch_size = len(batch_prefix_text)

                batch_output_text = self.greedy_until(batch_prefix_text, batch_until, max_length)
                output_text.extend(self.to_list(batch_output_text)[:batch_size])

            output = {
                'prefix_text': prefix_text,
                'until': until,
                'max_length': max_length,
                'output_text': output_text,
            }
            if self.config.logging:
                absl.logging.info(
                    '\n========= Output ========= \n'
                    + pprint.pformat(output) + '\n'
                )
        return output

    def serve_ready(self):
        return 'Ready!\n'

    def run(self):
        if self.config.pre_compile != '':
            if self.config.pre_compile == 'all':
                pre_compile = ['loglikelihood', 'generate', 'greedy_until']
            else:
                pre_compile = self.config.pre_compile.split(',')

            pre_compile_data = ['a' for _ in range(self.config.batch_size)]
            for task in pre_compile:
                if task == 'loglikelihood':
                    self.loglikelihood(pre_compile_data, pre_compile_data)
                    self.loglikelihood_rolling(pre_compile_data)
                elif task == 'generate':
                    self.generate(pre_compile_data, 1.0)
                elif task == 'greedy_until':
                    self.greedy_until(
                        pre_compile_data, pre_compile_data,
                        self.config.default_max_length
                    )
                else:
                    raise ValueError(f'Invalid precompile task: {task}!')

        self.app.run(host=self.config.host, port=self.config.port)

