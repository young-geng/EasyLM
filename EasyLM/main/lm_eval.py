import dataclasses
import pprint
from functools import partial
import os
import json
import urllib

import requests
from tqdm import tqdm, trange
import numpy as np
import wandb

import absl.app
import absl.flags
import absl.logging

from flax.traverse_util import flatten_dict
from lm_eval import evaluator, tasks
from lm_eval.base import LM

from ..utils import (
    WandBLogger, define_flags_with_default, get_user_flags, set_random_seed,
    load_pickle
)



FLAGS_DEF = define_flags_with_default(
    lm_server_url='http://localhost:5007/',
    tasks='wsc,piqa,winogrande,openbookqa,logiqa',
    shots=0,
    logger=WandBLogger.get_default_config(),
)
FLAGS = absl.flags.FLAGS


class LMEvalHarnessInterface(LM):

    def __init__(self, url):
        self.url = url

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
    logger = WandBLogger(
        config=FLAGS.logger, variant=get_user_flags(FLAGS, FLAGS_DEF)
    )
    model = LMEvalHarnessInterface(FLAGS.lm_server_url)
    task_list = FLAGS.tasks.split(',')
    results = evaluator.evaluate(
        model, tasks.get_task_dict(task_list), False, FLAGS.shots, None
    )
    logger.log(flatten_dict(results['results'], sep='/'))
    pprint.pprint(results)


if __name__ == "__main__":
    absl.app.run(main)
