import dataclasses
import pprint
from functools import partial
import re
import os
from threading import Lock
import urllib
import time
from typing import List, Optional

from pydantic import BaseModel
import absl.logging
from tqdm import tqdm, trange
import numpy as np
import mlxu
from ml_collections import ConfigDict
import uvicorn
from fastapi import FastAPI
import gradio as gr
import requests
from requests.exceptions import Timeout, ConnectionError


class LMInferenceRequest(BaseModel):
    prefix_text: Optional[List[str]] = None
    text: Optional[List[str]] = None
    until: Optional[List[str]] = None


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
        config.greedy_until_max_length = 5000
        config.chat_prepend_text = ''
        config.chat_user_prefix = ''
        config.chat_user_suffix = ''
        config.chat_lm_prefix = ''
        config.chat_lm_suffix = ''

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self.lock = Lock()
        self.app = FastAPI()
        self.app.post('/loglikelihood')(self.serve_loglikelihood)
        self.app.post('/loglikelihood-rolling')(self.serve_loglikelihood_rolling)
        self.app.post('/generate')(self.serve_generate)
        self.app.post('/greedy-until')(self.serve_greedy_until)
        self.app.get('/ready')(self.serve_ready)
        self.app = gr.mount_gradio_app(self.app, self.create_chat_app(), '/')

    @staticmethod
    def loglikelihood(prefix_text, text):
        raise NotImplementedError()

    @staticmethod
    def loglikelihood_rolling(text):
        raise NotImplementedError()

    @staticmethod
    def generate(text):
        raise NotImplementedError()

    @staticmethod
    def greedy_until(prefix_text, until, max_length):
        raise NotImplementedError()

    @staticmethod
    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    def serve_loglikelihood(self, data: LMInferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Log Likelihood Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )

            text = data.text
            if 'prefix_text' not in data:
                prefix_text = ['' for _ in text]
            else:
                prefix_text = data.prefix_text

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

    def serve_loglikelihood_rolling(self, data: LMInferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Log Likelihood Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )

            text = data.text
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

    def serve_generate(self, data: LMInferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Generate Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )
            prefix_text = data.prefix_text

            output_text = []
            for i in trange(0, len(prefix_text), self.config.batch_size, ncols=0):
                batch_prefix_text = prefix_text[i:i + self.config.batch_size]
                batch_size = len(batch_prefix_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_prefix_text.extend(['a' for _ in range(extra)])

                batch_output_text = self.generate(batch_prefix_text)
                output_text.extend(self.to_list(batch_output_text)[:batch_size])

            output = {
                'prefix_text': prefix_text,
                'output_text': output_text,
            }
            if self.config.logging:
                absl.logging.info(
                    '\n========= Output ========= \n'
                    + pprint.pformat(output) + '\n'
                )
        return output

    def serve_greedy_until(self, data: LMInferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Greedy Until Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )
            prefix_text = data.prefix_text
            until = data.until
            max_length = self.config.greedy_until_max_length

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

    def create_chat_app(self):
        with gr.Blocks() as gradio_chatbot:
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")
            context_state = gr.State('')

            def user_fn(user_message, history):
                return "", history + [[user_message, None]]

            def model_fn(history, context):
                response, context = self.process_chat(history[-1][0], context)
                history[-1][1] = response
                return history, context

            def clear_fn():
                return None, ''

            msg.submit(user_fn, [msg, chatbot], [msg, chatbot], queue=False).then(
                model_fn, [chatbot, context_state], [chatbot, context_state],
                queue=True
            )
            clear.click(clear_fn, None, [chatbot, context_state], queue=False)

        gradio_chatbot.queue(concurrency_count=1)
        return gradio_chatbot

    def process_chat(self, prompt, context):
        context = (
            context + self.config.chat_user_prefix
            + prompt + self.config.chat_user_suffix
            + self.config.chat_lm_prefix
        )
        response = self.generate([self.config.chat_prepend_text + context])[0]
        context = context + response + self.config.chat_lm_suffix
        return response, context

    def serve_ready(self):
        return 'Ready!\n'

    def run_server(self):
        if self.config.pre_compile != '':
            if self.config.pre_compile == 'all':
                pre_compile = ['loglikelihood', 'generate', 'greedy_until', 'chat']
            else:
                pre_compile = self.config.pre_compile.split(',')

            pre_compile_data = ['a' for _ in range(self.config.batch_size)]
            for task in pre_compile:
                if task == 'loglikelihood':
                    self.loglikelihood(pre_compile_data, pre_compile_data)
                    self.loglikelihood_rolling(pre_compile_data)
                elif task == 'generate':
                    self.generate(pre_compile_data)
                elif task == 'greedy_until':
                    self.greedy_until(
                        pre_compile_data, pre_compile_data,
                        self.config.greedy_until_max_length
                    )
                elif task == 'chat':
                    # Compile a batch 1 generate for chat
                    self.generate(['a'])
                else:
                    raise ValueError(f'Invalid precompile task: {task}!')

        uvicorn.run(self.app, host=self.config.host, port=self.config.port)

    def run(self):
        self.run_server()


class LMClient(object):
    """ A simple client for the LM server. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.url = 'http://localhost:5007'
        config.wait_for_ready = True
        config.dummy = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config=None):
        self.config = self.get_default_config(config)
        if self.config.wait_for_ready:
            self.wait_for_ready()

    def wait_for_ready(self):
        if self.config.dummy:
            return
        while True:
            try:
                requests.get(urllib.parse.urljoin(self.config.url, 'ready'))
                return
            except (Timeout, ConnectionError) as e:
                time.sleep(10)

    def loglikelihood(self, prefix, text):
        prefix, text = list(prefix), list(text)
        if self.config.dummy:
            return [-1.0 for _ in text], [False for _ in text]

        response = requests.post(
            urllib.parse.urljoin(self.config.url, 'loglikelihood'),
            json={'prefix_text': prefix, 'text': text}
        ).json()
        return response['log_likelihood'], response['is_greedy']

    def loglikelihood_rolling(self, text):
        text = list(text)
        if self.config.dummy:
            return [-1.0 for _ in text], [False for _ in text]
        response = requests.post(
            urllib.parse.urljoin(self.config.url, 'loglikelihood-rolling'),
            json={'text': text}
        ).json()
        return response['log_likelihood'], response['is_greedy']

    def greedy_until(self, prefix, until):
        prefix, until = list(prefix), list(until)
        if self.config.dummy:
            return until
        response = requests.post(
            urllib.parse.urljoin(self.config.url, 'greedy-until'),
            json={'prefix_text': prefix, 'until': until}
        ).json()
        return response['output_text']

    def generate(self, prefix):
        prefix = list(prefix)
        if self.config.dummy:
            return ['' for _ in prefix]
        response = requests.post(
            urllib.parse.urljoin(self.config.url, 'generate'),
            json={'prefix_text': prefix}
        ).json()
        return response['output_text']
