# Evaluating Language Models
EasyLM has builtin support for evaluating language models on a variety of tasks.
Once the trained language model is served with LMServer, it can be evaluated
against various benchmarks in few-shot and zero-shot settings.

## LM Evaluation Harness
EasyLM comes with builtin support for [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness),
which can evaluate the language model on a variety of tasks. For example,
you can use the following command to evaluate the langauge model served with
the HTTP server:

```shell
python -m EasyLM.scripts.lm_eval_harness \
    --lm_client.url='http://localhost:5007/' \
    --tasks='wsc,piqa,winogrande,openbookqa,logiqa' \
    --shots=0
```

The `lm_eval_harness` script supports the following commnad line options:
* `tasks`: a comma separated list of tasks to evaluate the language model on.
  The supported tasks are listed in the
  [lm-eval-harness task table](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md)
* `shots`: the number of shots to use for the evaluation.
* `batch_size`: the batch size to use for each http request. Too large a batch
  size may cause the request to time out. Default to 1.
* `lm_client`: the configurations for LMClient. See [the LMClient documentation](serving.md)
  for more details.
* `logger`: the configurations for the logger. See [the logger documentation](logger.md)
  for more details.

## Evaluating on MMLU
The served langauge model can also be evaluated with the [MMLU](https://github.com/hendrycks/test)
benchmark. In order to run the evaluation, you'll need to use [my fork of MMLU](https://github.com/young-geng/mmlu_easylm) which supports EasyLM LMServer.

```shell
git clone https://github.com/young-geng/mmlu_easylm.git
cd mmlu_easylm
python evaluate_easylm.py \
    --name='llama' \
    --lm_server_url='http://localhost:5007' \
    --ntrain=5
```


