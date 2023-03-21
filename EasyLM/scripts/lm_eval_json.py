import json
import mlxu
from EasyLM.serving import LMClient


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    input_file='',
    output_file='',
    prefix_field='prefix',
    text_field='text',
    until_field='until',
    eval_type='loglikelihood',
    lm_client=LMClient.get_default_config(),
)


def main(argv):
    lm_client = LMClient(FLAGS.lm_client)
    with mlxu.open_file(FLAGS.input_file, 'r') as fin:
        input_data = json.load(fin)

    if FLAGS.eval_type == 'loglikelihood':
        prefix = input_data[FLAGS.prefix_field]
        text = input_data[FLAGS.text_field]
        loglikelihoods, is_greedys = lm_client.loglikelihood(prefix, text)
        output_data = {
            'loglikelihood': loglikelihoods,
            'is_greedy': is_greedys,
        }
    elif FLAGS.eval_type == 'loglikelihood_rolling':
        text = input_data[FLAGS.text_field]
        loglikelihoods, is_greedys = lm_client.loglikelihood_rolling(text)
        output_data = {
            'loglikelihood': loglikelihoods,
            'is_greedy': is_greedys,
        }
    elif FLAGS.eval_type == 'greedy_until':
        prefix = input_data[FLAGS.prefix_field]
        until = input_data[FLAGS.until_field]
        output_data = {'output_text': lm_client.greedy_until(prefix, until)}
    elif FLAGS.eval_type == 'generate':
        prefix = input_data[FLAGS.prefix_field]
        output_data = {'output_text': lm_client.generate(prefix)}
    else:
        raise ValueError(f'Unknown eval_type: {FLAGS.eval_type}')

    with mlxu.open_file(FLAGS.output_file, 'w') as fout:
        json.dump(output_data, fout)


if __name__ == "__main__":
    mlxu.run(main)
