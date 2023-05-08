import dataclasses
import pprint
from functools import partial
import re
import os
from threading import Lock
import urllib
import time
from typing import List, Optional, Union

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


class InferenceRequest(BaseModel):
    prefix_text: Optional[List[str]] = None
    text: Optional[List[str]] = None
    until: Optional[Union[List[str], List[List[str]]]] = None
    temperature: Optional[float] = None


class ChatRequest(BaseModel):
    prompt: str
    context: str = ''
    temperature: Optional[float] = None


class LMServer(object):
    """ HTTP server for serving langauge models. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.host = '0.0.0.0'
        config.port = 5007
        config.batch_size = 1
        config.logging = False
        config.pre_compile = 'loglikelihood'
        config.default_temperature = 1.0
        config.greedy_until_max_length = 5000
        config.prepend_to_prefix = ''
        config.append_to_prefix = ''
        config.prepend_to_text = ''
        config.append_to_text = ''
        config.chat_prepend_text = ''
        config.chat_user_prefix = ''
        config.chat_user_suffix = ''
        config.chat_lm_prefix = ''
        config.chat_lm_suffix = ''
        config.notes = ''

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
        self.app.post('/chat')(self.serve_chat)
        self.app.get('/ready')(self.serve_ready)
        self.app = gr.mount_gradio_app(self.app, self.create_chat_app(), '/')

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

    def serve_ready(self):
        return 'Ready!\n'

    def serve_loglikelihood(self, data: InferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Log Likelihood Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )

            if data.prefix_text is None:
                data.prefix_text = ['' for _ in data.text]

            prefix_text = [
                self.config.prepend_to_prefix + p + self.config.append_to_prefix
                for p in data.prefix_text
            ]
            text = [
                self.config.prepend_to_text + t + self.config.append_to_text
                for t in data.text
            ]

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
                'prefix_text': data.prefix_text,
                'text': data.text,
                'log_likelihood': log_likelihood,
                'is_greedy': is_greedy,
            }
            if self.config.logging:
                absl.logging.info(
                '\n========= Output ========= \n'
                + pprint.pformat(output) + '\n'
            )

        return output

    def serve_loglikelihood_rolling(self, data: InferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Log Likelihood Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )

            text = [
                self.config.prepend_to_text + t + self.config.append_to_text
                for t in data.text
            ]
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
                'text': data.text,
                'log_likelihood': log_likelihood,
                'is_greedy': is_greedy,
            }
            if self.config.logging:
                absl.logging.info(
                '\n========= Output ========= \n'
                + pprint.pformat(output) + '\n'
            )

        return output

    def serve_generate(self, data: InferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Generate Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )
            prefix_text = [
                self.config.prepend_to_prefix + p + self.config.append_to_prefix
                for p in data.prefix_text
            ]

            if data.temperature is None:
                data.temperature = self.config.default_temperature

            output_text = []
            for i in trange(0, len(prefix_text), self.config.batch_size, ncols=0):
                batch_prefix_text = prefix_text[i:i + self.config.batch_size]
                batch_size = len(batch_prefix_text)

                if batch_size < self.config.batch_size:
                    extra = self.config.batch_size - batch_size
                    batch_prefix_text.extend(['a' for _ in range(extra)])

                batch_output_text = self.generate(
                    batch_prefix_text,
                    temperature=data.temperature,
                )
                output_text.extend(self.to_list(batch_output_text)[:batch_size])

            output = {
                'prefix_text': data.prefix_text,
                'output_text': output_text,
                'temperature': data.temperature,
            }
            if self.config.logging:
                absl.logging.info(
                    '\n========= Output ========= \n'
                    + pprint.pformat(output) + '\n'
                )
        return output

    def serve_greedy_until(self, data: InferenceRequest):
        with self.lock:
            if self.config.logging:
                absl.logging.info(
                    '\n========= Serving Greedy Until Request ========= \n'
                    + pprint.pformat(data) + '\n'
                )
            prefix_text = [
                self.config.prepend_to_prefix + p + self.config.append_to_prefix
                for p in data.prefix_text
            ]
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
                'prefix_text': data.prefix_text,
                'until': data.until,
                'max_length': max_length,
                'output_text': output_text,
            }
            if self.config.logging:
                absl.logging.info(
                    '\n========= Output ========= \n'
                    + pprint.pformat(output) + '\n'
                )
        return output

    def process_chat(self, prompt, context, temperature):
        context = (
            context + self.config.chat_user_prefix
            + prompt + self.config.chat_user_suffix
            + self.config.chat_lm_prefix
        )
        response = self.generate(
            [self.config.chat_prepend_text + context],
            temperature=float(temperature),
        )[0]
        context = context + response + self.config.chat_lm_suffix
        return response, context

    def serve_chat(self, data: ChatRequest):
        if data.temperature is None:
            data.temperature = self.config.default_temperature
        response, context = self.process_chat(
            data.prompt, data.context,
            temperature=data.temperature,
        )
        return {
            'response': response,
            'context': context,
            'temperature': data.temperature,
        }

    def create_chat_app(self):
        with gr.Blocks(analytics_enabled=False, title='EasyLM Chat') as gradio_chatbot:
            gr.Markdown('# Chatbot Powered by [EasyLM](https://github.com/young-geng/EasyLM)')
            gr.Markdown(self.config.notes)
            chatbot = gr.Chatbot(label='Chat history')
            msg = gr.Textbox(
                placeholder='Type your message here...',
                show_label=False
            )
            with gr.Row():
                send = gr.Button('Send')
                regenerate = gr.Button('Regenerate', interactive=False)
                clear = gr.Button('Reset')

            temp_slider = gr.Slider(
                label='Temperature', minimum=0, maximum=2.0,
                value=self.config.default_temperature
            )

            context_state = gr.State(['', ''])

            def user_fn(user_message, history, context):
                return {
                    msg: gr.update(value='', interactive=False),
                    clear: gr.update(interactive=False),
                    send: gr.update(interactive=False),
                    regenerate: gr.update(interactive=False),
                    chatbot: history + [[user_message, None]],
                    context_state: [context[1], context[1]],
                }

            def model_fn(history, context, temperature):
                history[-1][1], new_context = self.process_chat(
                    history[-1][0], context[0], temperature
                )
                return {
                    msg: gr.update(value='', interactive=True),
                    clear: gr.update(interactive=True),
                    send: gr.update(interactive=True),
                    chatbot: history,
                    context_state: [context[0], new_context],
                    regenerate: gr.update(interactive=True),
                }

            def regenerate_fn():
                return {
                    msg: gr.update(value='', interactive=False),
                    clear: gr.update(interactive=False),
                    send: gr.update(interactive=False),
                    regenerate: gr.update(interactive=False),
                }

            def clear_fn():
                return {
                    chatbot: None,
                    msg: '',
                    context_state: ['', ''],
                    regenerate: gr.update(interactive=False),
                }

            msg.submit(
                user_fn,
                inputs=[msg, chatbot, context_state],
                outputs=[msg, clear, send, chatbot, context_state, regenerate],
                queue=False
            ).then(
                model_fn,
                inputs=[chatbot, context_state, temp_slider],
                outputs=[msg, clear, send, chatbot, context_state, regenerate],
                queue=True
            )
            send.click(
                user_fn,
                inputs=[msg, chatbot, context_state],
                outputs=[msg, clear, send, chatbot, context_state, regenerate],
                queue=False
            ).then(
                model_fn,
                inputs=[chatbot, context_state, temp_slider],
                outputs=[msg, clear, send, chatbot, context_state, regenerate],
                queue=True
            )
            regenerate.click(
                regenerate_fn,
                inputs=None,
                outputs=[msg, clear, send, regenerate],
                queue=False
            ).then(
                model_fn,
                inputs=[chatbot, context_state, temp_slider],
                outputs=[msg, clear, send, chatbot, context_state, regenerate],
                queue=True
            )
            clear.click(
                clear_fn,
                inputs=None,
                outputs=[chatbot, msg, context_state, regenerate],
                queue=False
            )

        gradio_chatbot.queue(concurrency_count=1)
        return gradio_chatbot

    def run(self):
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
                    self.generate(pre_compile_data, 1.0)
                elif task == 'greedy_until':
                    self.greedy_until(
                        pre_compile_data, pre_compile_data,
                        self.config.greedy_until_max_length
                    )
                elif task == 'chat':
                    self.process_chat('a', 'a', 1.0)
                else:
                    raise ValueError(f'Invalid precompile task: {task}!')

        uvicorn.run(self.app, host=self.config.host, port=self.config.port)


class LMClient(object):
    """ A simple client for the LM server. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.url = 'http://localhost:5007'
        config.batch_size = 1
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

    @staticmethod
    def batched(iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def loglikelihood(self, prefix, text):
        prefix, text = list(prefix), list(text)
        if self.config.dummy:
            return [-1.0 for _ in text], [False for _ in text]

        log_likelihood = []
        is_greedy = []

        batched_iterator = list(zip(
            self.batched(prefix, self.config.batch_size),
            self.batched(text, self.config.batch_size)
        ))
        for batch_prefix, batch_text in tqdm(batched_iterator, ncols=0):
            response = requests.post(
                urllib.parse.urljoin(self.config.url, 'loglikelihood'),
                json={'prefix_text': batch_prefix, 'text': batch_text}
            ).json()
            log_likelihood.extend(response['log_likelihood'])
            is_greedy.extend(response['is_greedy'])

        return log_likelihood, is_greedy

    def loglikelihood_rolling(self, text):
        text = list(text)
        if self.config.dummy:
            return [-1.0 for _ in text], [False for _ in text]

        log_likelihood = []
        is_greedy = []
        batched_iterator = list(self.batched(text, self.config.batch_size))
        for batch_text in tqdm(batched_iterator, ncols=0):
            response = requests.post(
                urllib.parse.urljoin(self.config.url, 'loglikelihood-rolling'),
                json={'text': batch_text}
            ).json()
            log_likelihood.extend(response['log_likelihood'])
            is_greedy.extend(response['is_greedy'])
        return log_likelihood, is_greedy

    def greedy_until(self, prefix, until):
        prefix, until = list(prefix), list(until)
        if self.config.dummy:
            results = []
            for u in until:
                if isinstance(u, str):
                    results.append('dummy text ' + u)
                else:
                    results.append('dummy text ' + u[0])
            return results

        batched_iterator = list(zip(
            self.batched(prefix, self.config.batch_size),
            self.batched(until, self.config.batch_size),
        ))
        output_text = []
        for batch_prefix, batch_until in tqdm(batched_iterator, ncols=0):
            response = requests.post(
                urllib.parse.urljoin(self.config.url, 'greedy-until'),
                json={'prefix_text': batch_prefix, 'until': batch_until}
            ).json()
            output_text.extend(response['output_text'])
        return output_text

    def generate(self, prefix, temperature=None):
        prefix = list(prefix)
        if self.config.dummy:
            return ['' for _ in prefix]

        output_text = []
        batched_iterator = list(self.batched(prefix, self.config.batch_size))
        for batch_prefix in tqdm(batched_iterator, ncols=0):
            response = requests.post(
                urllib.parse.urljoin(self.config.url, 'generate'),
                json={
                    'prefix_text': batch_prefix,
                    'temperature': temperature,
                }
            ).json()
            output_text.extend(response['output_text'])
        return output_text

    def chat(self, prompt, context, temperature=None):
        if self.config.dummy:
            return ''
        response = requests.post(
            urllib.parse.urljoin(self.config.url, 'chat'),
            json={
                'prompt': prompt,
                'context': context,
                'temperature': temperature,
            }
        ).json()
        return response['response'], response['context']
