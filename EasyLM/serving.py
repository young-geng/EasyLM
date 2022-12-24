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

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.name = 'lm_server'
        config.host = '0.0.0.0'
        config.port = 5007
        config.logging = True
        config.pre_compile = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, loglikelihood_fn, generate_fn):
        self.config = self.get_default_config(config)
        self.loglikelihood_fn = loglikelihood_fn
        self.generate_fn = generate_fn

        self.lock = Lock()
        self.app = Flask(self.config.name)
        self.app.post('/loglikelihood')(self.loglikelihood)
        self.app.post('/generate')(self.generate)

    def loglikelihood(self):
        with self.lock:
            data = request.get_json()
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Log Likelihood Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )

            input_text = data['input_text']
            log_likelihood = self.loglikelihood_fn(input_text)

            if isinstance(log_likelihood, np.ndarray):
                log_likelihood = log_likelihood.tolist()

            output = {'log_likelihood': log_likelihood}
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
            input_text = data['input_text']
            temperature = data.get('temperature', 1.0)
            output_text = self.generate_fn(input_text, temperature)
            output = {'output_text': output_text}
            if self.config.logging:
                absl.logging.info(
                    '\n========= Output ========= \n'
                    + pprint.pformat(output) + '\n'
                )
        return output

    def run(self):
        if self.config.pre_compile:
            self.loglikelihood_fn(['pre_compile'])
            self.generate_fn(['pre_compile'], 1.0)
        self.app.run(host=self.config.host, port=self.config.port)

