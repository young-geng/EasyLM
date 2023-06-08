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
    limit=0,
    write_out=False,
    lm_client=LMClient.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
)


class LMEvalHarnessInterface(LM):

    def __init__(self, lm_client):
        self.lm_client = lm_client

    def greedy_until(self, inputs):
        prefix, until = zip(*inputs)
        return self.lm_client.greedy_until(prefix, until)

    def loglikelihood_rolling(self, inputs):
        loglikelihood, is_greedy = self.lm_client.loglikelihood_rolling(inputs)
        return list(zip(loglikelihood, is_greedy))

    def loglikelihood(self, inputs):
        prefix, text = zip(*inputs)
        loglikelihood, is_greedy = self.lm_client.loglikelihood(prefix, text)
        return list(zip(loglikelihood, is_greedy))


def main(argv):
    logger = mlxu.WandBLogger(
        config=FLAGS.logger, variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    )
    model = LMEvalHarnessInterface(LMClient(FLAGS.lm_client))
    task_list = FLAGS.tasks.split(',')
    results = evaluator.evaluate(
        model, tasks.get_task_dict(task_list), False, FLAGS.shots,
        limit=None if FLAGS.limit <= 0 else FLAGS.limit,
        write_out=FLAGS.write_out,
    )
    logger.log(flatten_dict(results['results'], sep='/'))
    pprint.pprint(results)


if __name__ == "__main__":
    mlxu.run(main)
