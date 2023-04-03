# Serving and Client
EasyLM provides an HTTP server and a client for serving and querying language
models. The server and client are implemented in [serving.py](/EasyLM/serving.py).
The HTTP server serves multiple endpoints as well as a chat web UI for querying
the language model. These endpoints can be used to interact with the language
model or perform batch evaluation.


## LMServer Interface
The `LMServer` class implements the HTTP server. Each language model serving
script should inherit from this class and implement the following static methods:

* `loglikelihood(prefix_text, text)`: given a list of prefix text strings and a
  list of text strings, return the loglikelihood of the text strings given the
  prefix text strings. The prefix text strings do not contribute to the
  total loglikelihood. This static method returns a pair of lists, where the
  first list contains the loglikelihoods of the text strings and the second list
  contains whether the text strings match the greedy decoding choice with the
  maximum log likelihood.
* `loglikelihood_rolling(text)`: computes the log likelihood of the text strings,
  where the text strings might be longer than the maximum sequence length of the
  language model. If the text strings are longer than the maximum sequence length,
  the log likelihood is computed using a window. This method returns a pair of
  lists, where the first list contains the loglikelihoods of the text strings and
  the second list contains whether the text strings match the greedy decoding.
* `generate(prefix_text, temperature)`: given a list of prefix text strings and
  a temperature value, generate a list of strings. This method returns the list
  of generated strings.
* `greedy_until(prefix_text, until, max_length)`: given a list of prefix text
  strings, a list of until strings, and a maximum length, generate a list of
  strings greedily. The generated strings will be generated until the until strings are
  generated, or the maximum length is reached. This method returns the list of
  generated strings.

These static methods are called by the HTTP server to serve the endpoints. These
methods are defined largely by the [Language Model Evaluation Harness
](https://github.com/EleutherAI/lm-evaluation-harness) library, which is used by
EasyLM to evaluate the served language models.


## LMServer Endpoints and LMCient
The `LMServer` class implements the following endpoints for querying the language
model with HTTP requests. These endpoints can be queried by sending a JSON
dictionary using the POST method. The following fields are used for each endpoint:

#### `/loglikelihood`
The input JSON dictionary should contain the following fields:
* `prefix_text`: a list of prefix text strings.
* `text`: a list of text strings.

The output JSON dictionary contains the following fields:
* `loglikelihood`: a list of loglikelihoods of the text strings given the prefix
  text strings.
* `is_greedy`: a list of booleans indicating whether the text strings match the
  greedy decoding choice with the maximum log likelihood.


#### `/serve_loglikelihood_rolling`
The input JSON dictionary should contain the following fields:
* `text`: a list of text strings.

The output JSON dictionary contains the following fields:
* `loglikelihood`: a list of loglikelihoods of the text strings.
* `is_greedy`: a list of booleans indicating whether the text strings match the
  greedy decoding choice with the maximum log likelihood.


#### `/generate`
The input JSON dictionary should contain the following fields:
* `prefix_text`: a list of prefix text strings.
* `temperature`: a temperature value.

The output JSON dictionary contains the following fields:
* `output_text`: a list of generated text strings.


#### `/greedy_until`
The input JSON dictionary should contain the following fields:
* `prefix_text`: a list of prefix text strings.
* `until`: a list of until strings.

The output JSON dictionary contains the following fields:
* `output_text`: a list of generated text strings.


#### `/chat`
The chat endpoint is intended to be used for a dialogue language model. The input
JSON dictionary should contain the following fields:
* `prompt`: a prompt string.
* `context`: a context string. Can be empty for the first query.
* `temperature`: a temperature value.

The output JSON dictionary contains the following fields:
* `response`: a model response string.
* `context`: the updated context string containing the chat history. This is
  used for the next round of dialogue.

### Chat UI
For interacting with a dialogue language model over the web UI, simply navigate
to the root of the HTTP server. The chat UI will be served at the root URL.


### LMCient
The `LMClient` class implements a client for querying the served language model.
The python methods of this class are similar to the endpoints of the HTTP server.


## LMServer Options
The `LMServer` class implements the following command line options:
* `host`: the host ip address to serve the HTTP server.
* `port`: the port to serve the HTTP server.
* `batch_size`: the batch size for serving the language model.
* `logging`: whether to log the requests to the HTTP server.
* `pre_compile`: a command separated list of endpoints to trigger JAX compilation
  before serving the language model. This is useful for speeding up the first
  request to the language model. The following endpoints are supported:
  `loglikelihood`, `generate`, `greedy_until`, `chat`, or `all` for all endpoints.
* `default_temperature`: the default temperature for the `generate` endpoint.
* `greedy_until_max_length`: the maximum length for the `greedy_until` endpoint.
* `prepend_to_prefix`: a string to prepend to the prefix text strings for the
  `loglikelihood` and `generate` and `greedy_until` endpoints.
* `append_to_prefix`: a string to append to the prefix text strings for the
  `loglikelihood` and `generate` and `greedy_until` endpoints.
* `prepend_to_text`: a string to prepend to the text strings for the `loglikelihood`
   endpoint.
* `append_to_text`: a string to append to the text strings for the `loglikelihood`
   endpoint.
* `chat_prepend_text`: a string to prepend to the context strings for the `chat`
   endpoint.
* `chat_user_prefix`: a string to prepend to the user input strings for the `chat`
   endpoint.
* `chat_user_suffix`: a string to append to the user input strings for the `chat`
   endpoint.
* `chat_lm_prefix`: a string to prepend to the model response strings for the `chat`
   endpoint.
* `chat_lm_suffix`: a string to append to the model response strings for the `chat`
* `notes`: a string to display on the chat UI.


## LMClient Options
The `LMClient` class implements the following command line options:
* `url`: the base URL of the HTTP server.
* `wait_for_ready`: whether to wait for the HTTP server to be ready before
  sending requests.
* `dummy`: whether to use a dummy language model for debugging. If set to True,
  the LMCient will always return some fixed results.