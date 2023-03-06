# This script runs lm_eval_harness evaluations against a served language model.
# Typically, you need to run a language model server first, e.g.:
#    python -m EasyLM.models.gptj.gptj_serve ...

import dataclasses
import pprint
from functools import partial
import os
import json
import urllib
import time

import requests
from requests.exceptions import Timeout, ConnectionError
from tqdm import tqdm, trange
import numpy as np
import mlxu

from flax.traverse_util import flatten_dict
from lm_eval import evaluator, tasks
from lm_eval.base import LM


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    dummy_lm=False,
    lm_server_url='http://localhost:5007/',
    tasks='wsc,piqa,winogrande,openbookqa,logiqa',
    shots=0,
    wait_for_ready=True,
    logger=mlxu.WandBLogger.get_default_config(),
)


class DummyLM(LM):

    def wait_for_ready(self):
        pass

    def greedy_until(self, inputs):
        return [until for _, until in inputs]

    def loglikelihood_rolling(self, inputs):
        return [(-20.0, True) for _ in inputs]

    def loglikelihood(self, inputs):
        return [(-20.0, True) for _ in inputs]


class LMEvalHarnessInterface(LM):

    def __init__(self, url):
        self.url = url

    def wait_for_ready(self):
        while True:
            try:
                requests.get(urllib.parse.urljoin(self.url, 'ready'))
                return
            except (Timeout, ConnectionError) as e:
                time.sleep(10)

    def greedy_until(self, inputs):
        prefix, until = zip(*inputs)
        prefix = list(prefix)
        until = list(until)
        response = requests.post(
            urllib.parse.urljoin(self.url, 'greedy-until'),
            json={'prefix_text': prefix, 'until': until}
        ).json()
        return list(response['output_text'])

    def loglikelihood_rolling(self, inputs):
        text = list(inputs)
        response = requests.post(
            urllib.parse.urljoin(self.url, 'loglikelihood-rolling'),
            json={'text': text}
        ).json()
        return list(zip(response['log_likelihood'], response['is_greedy']))

    def loglikelihood(self, inputs):
        prefix, text = zip(*inputs)
        prefix = list(prefix)
        text = list(text)
        response = requests.post(
            urllib.parse.urljoin(self.url, 'loglikelihood'),
            json={'prefix_text': prefix, 'text': text}
        ).json()
        return list(zip(response['log_likelihood'], response['is_greedy']))


def main(argv):
    logger = mlxu.WandBLogger(
        config=FLAGS.logger, variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    )
    if FLAGS.dummy_lm:
        model = DummyLM()
    else:
        model = LMEvalHarnessInterface(FLAGS.lm_server_url)
    if FLAGS.wait_for_ready:
        model.wait_for_ready()
    task_list = FLAGS.tasks.split(',')
    results = evaluator.evaluate(
        model, tasks.get_task_dict(task_list), False, FLAGS.shots, None
    )
    logger.log(flatten_dict(results['results'], sep='/'))
    pprint.pprint(results)


if __name__ == "__main__":
    mlxu.run(main)
