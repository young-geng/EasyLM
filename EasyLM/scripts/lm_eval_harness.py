# This script runs lm_eval_harness evaluations against a served language model.
# Typically, you need to run a language model server first, e.g.:
#    python -m EasyLM.models.gptj.gptj_serve ...

import dataclasses
import pprint
from functools import partial
import os
from tqdm import tqdm, trange
import numpy as np
import mlxu

from flax.traverse_util import flatten_dict
from lm_eval import evaluator, tasks
from lm_eval.base import LM

from EasyLM.serving import LMClient


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    tasks='wsc,piqa,winogrande,openbookqa,logiqa',
    shots=0,
    batch_size=1,
    lm_client=LMClient.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
)


class LMEvalHarnessInterface(LM):

    def __init__(self, lm_client):
        self.lm_client = lm_client

    def greedy_until(self, inputs):
        results = []
        batched_inputs = list(batched(inputs, FLAGS.batch_size))
        for batch in tqdm(batched_inputs, desc='greedy_until', ncols=0):
            prefix, until = zip(*batch)
            results.extend(self.lm_client.greedy_until(prefix, until))
        return results

    def loglikelihood_rolling(self, inputs):
        loglikelihoods, is_greedys = [], []
        batched_inputs = list(batched(inputs, FLAGS.batch_size))
        for batch in tqdm(batched_inputs, desc='loglikelihood_rolling', ncols=0):
            ll, greedy = self.lm_client.loglikelihood_rolling(batch)
            loglikelihoods.extend(ll)
            is_greedys.extend(greedy)
        return list(zip(loglikelihoods, is_greedys))

    def loglikelihood(self, inputs):
        loglikelihoods, is_greedys = [], []
        batched_inputs = list(batched(inputs, FLAGS.batch_size))
        for batch in tqdm(batched_inputs, desc='loglikelihood', ncols=0):
            prefix, text = zip(*batch)
            ll, greedy = self.lm_client.loglikelihood(prefix, text)
            loglikelihoods.extend(ll)
            is_greedys.extend(greedy)
        return list(zip(loglikelihoods, is_greedys))


def batched(iterator, batch_size):
    batch = []
    for example in iterator:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def main(argv):
    logger = mlxu.WandBLogger(
        config=FLAGS.logger, variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    )
    model = LMEvalHarnessInterface(LMClient(FLAGS.lm_client))
    task_list = FLAGS.tasks.split(',')
    results = evaluator.evaluate(
        model, tasks.get_task_dict(task_list), False, FLAGS.shots, None
    )
    logger.log(flatten_dict(results['results'], sep='/'))
    pprint.pprint(results)


if __name__ == "__main__":
    mlxu.run(main)
