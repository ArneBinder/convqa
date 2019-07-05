# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ast
import json
import logging
import os
import random
import sys
import time
import traceback
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
from tqdm import tqdm
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, Response, request

from train import MODELS, build_input_from_segments, MARKER_BACKGROUND, MARKER_SPEAKER1, MARKER_SPEAKER2
from utils import get_dataset_personalities, download_pretrained_model, create_wikipedia_context_fetcher

endpoint = Flask(__name__, static_url_path='')
#cors = CORS(endpoint)

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(context, history, tokenizer, model, args, current_output=None):
    max_sequence_length = args.max_sequence_length if args.max_sequence_length > 0 else model.config.n_ctx
    assert max_sequence_length <= model.config.n_ctx, 'max_sequence_length [%i] was set to a value higher than ' \
                                                      'supported by the model (config.n_ctx [%i]). Please use a lower ' \
                                                      'value or do not set it [-1] to use the highest supported one.' \
                                                      % (max_sequence_length, model.config.n_ctx)
    special_tokens_ids = tokenizer.special_tokens.values()
    background = tokenizer.special_tokens[MARKER_BACKGROUND]
    speaker1 = tokenizer.special_tokens[MARKER_SPEAKER1]
    speaker2 = tokenizer.special_tokens[MARKER_SPEAKER2]
    #logger.debug('expected sequence length (without prediction): %i; max_allowed: %i (inclusive prediction)'
    #             % (len(list(chain(*(context + history)))) + len(history) + 1, max_sequence_length))
    if current_output is None:
        current_output = []
    for i in range(args.max_length):
        # TODO: adapt changes of build_input_from_segments (expects list of tuples: context and history. Check that!)
        instance, sequence = build_input_from_segments(context=[(background, context)],
                                                       history=[(speaker2 if (len(history) - i) % 2 else speaker1, h) for i, h in enumerate(history)],
                                                       reply=(speaker1, current_output), tokenizer=tokenizer, eos=None,
                                                       max_sequence_length=max_sequence_length)
        l_trunc = len(list(chain(*sequence))) - len(instance['input_ids'])
        assert l_trunc <= 0, 'The sequence was truncated. Please provide less context + history + question!'

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@endpoint.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def parse_params(params, prev={}):
    result = prev
    for param in params:
        v_ = params[param]
        try:
            v = ast.literal_eval(v_)
        except ValueError:
            v = v_
        except SyntaxError:
            v = v_
            logging.warning('Syntax error while parsing "%s". Assume it is a string.' % v_)
        result[param] = v
    return result


def get_params():
    data = request.data.decode("utf-8")
    params = {}
    if data != "":
        params = json.loads(data)
    params = parse_params(request.args, params)
    params = parse_params(request.form, params)
    if request.headers.environ['HTTP_ACCEPT'] != '*/*':
        params['HTTP_ACCEPT'] = request.headers.environ['HTTP_ACCEPT']

    return params


@endpoint.route("/hello_world")
def hello_world():
    return "Hello World!"


@endpoint.route("/ask", methods=['GET', 'POST'])
def ask():
    try:
        start = time.time()
        logging.info('prediction requested')
        params = get_params()
        logger.debug(json.dumps(params, indent=2))

        history = params.get('history', [])
        user_input = params['user_input']

        # create required format of context: dict with entry_id -> list of sentences (strings)
        if isinstance(params.get('context', None), str):
            params['context'] = {'user': params['context']}

        context = params.get('context', None)
        if context is None:
            assert context_fetcher is not None, 'No context fetcher initialized (requires a spacy model). Please provide a context with every request.'
            params['context'] = context_fetcher(' '.join(history + [user_input]))
        elif context_fetcher is not None and not params.get('dont_refetch', False):
            params['context'] = context_fetcher(' '.join(history + [user_input]), context)

        history.append(user_input)
        context_encoded = [tokenizer.encode(article) for article in params['context'].values()]
        history_encoded = [tokenizer.encode(utterance) for utterance in history]
        with torch.no_grad():
            out_ids = sample_sequence(context=context_encoded, history=history_encoded, tokenizer=tokenizer,
                                      model=model, args=args)
        history_encoded.append(out_ids)
        history_encoded = history_encoded[-(2 * args.max_history + 1):]
        params['prediction'] = tokenizer.decode(out_ids, skip_special_tokens=True)
        params['history'] = [tokenizer.decode(utterance) for utterance in history_encoded]
        logger.debug('predicted:\n%s' % params['prediction'])

        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(params)
        response = Response(json_data, mimetype=return_type)

        logger.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        raise InvalidUsage('%s: %s @line %s in %s' % (type(e).__name__, str(e), exc_tb.tb_lineno, fname))
    return response


def init():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt", help="Model type, one of: %s" % ', '.join(MODELS.keys()))
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_sequence_length", type=int, default=-1, help="If set, use this to manually restrict the sequence length. "
                                                                            "This might be helpful to save resources (memory). "
                                                                            "If not set, this is looked up from the model config (n_ctx value).")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--start_endpoint", action='store_true', help="Start a flask endpoint")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="to allow automatic sentence splitting for flask endpoint")
    parser.add_argument("--coqa_file", type=str, default="", help="path to a file in the CoQA dataset format containing question where the answers will be predicted")
    parser.add_argument("--prediction_out", type=str, default="", help="path to a file to save the predictions")
    parser.add_argument("--wikipedia_dump", type=str, default="", help="path to a pickle file containing a dict: "
                                                                       "wikipedia cuid -> {'text': ['This is a sentence.', 'This is another sentence.']}")

    args = parser.parse_args()

    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    if args.model not in MODELS:
        raise NotImplementedError('model "%s" not implemented. use one of %s' % (args.model, str(MODELS.keys)))
    tokenizer_class, _, model_class = MODELS[args.model]

    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    return tokenizer, model, args


def run(tokenizer, model, args):
    if args.start_endpoint:
        logger.info('Starting the API')
        endpoint.run(host='0.0.0.0', port=5000)
    elif args.coqa_file:
        logger.info('predict answers for CoQA file: %s ...' % args.coqa_file)
        data = json.load(open(args.coqa_file))
        assert sentencizer is not None, 'No sentencizer initialized (requires a spacy model). This is required to process a CoQA dataset file.'
        predictions = []
        n_errors = 0
        n_total = 0
        for instance in tqdm(data['data']):
            context_sents = sentencizer(instance['story'])
            context_encoded = [tokenizer.encode(sentence) for sentence in context_sents]
            history_encoded = []
            for question in instance['questions']:
                n_total += 1
                question_text = question['input_text']
                history_encoded.append(tokenizer.encode(question_text))
                with torch.no_grad():
                    try:
                        out_ids = sample_sequence(context=context_encoded, history=history_encoded, tokenizer=tokenizer,
                                                  model=model, args=args)
                        history_encoded.append(out_ids)
                        history_encoded = history_encoded[-(2 * args.max_history + 1):]
                        answer_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                    except AssertionError as e:
                        logger.warning('ERROR (id: %s turn_id: %s): %s' % (instance['id'], question['turn_id'], e))
                        del history_encoded[-1]
                        answer_text = 'NONE'
                        n_errors += 1

                predictions.append({
                    'id': instance['id'],
                    'turn_id': question['turn_id'],
                    'answer': answer_text
                })
        logger.info('%i of %i predictions failed' % (n_errors, n_total))
        out_fn = args.prediction_out or os.path.join(args.model_checkpoint, 'predictions.json')
        logger.info('write predictions to: %s ...' % out_fn)
        json.dump(predictions, open(out_fn, 'w'), indent=2)
    else:
        logger.info("Sample a personality")
        personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
        personality = random.choice(personalities)
        history_encoded = []
        logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
        while True:
            raw_text = input(">>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input(">>> ")
            history_encoded.append(tokenizer.encode(raw_text))
            with torch.no_grad():
                out_ids = sample_sequence(context=personality, history=history_encoded, tokenizer=tokenizer,
                                          model=model, args=args)
            history_encoded.append(out_ids)
            history_encoded = history_encoded[-(2*args.max_history+1):]
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            print(out_text)


def create_sentencizer(spacy_model='en_core_web_sm'):
    import spacy
    nlp = spacy.load(spacy_model)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    #sentencizer = lambda s: [sent.text for sent in nlp(s.strip(), disable=['parser', 'tagger', 'ner']).sents]
    def sentencizer(s):
        sents = []
        for sent in nlp(s.strip(), disable=['parser', 'tagger', 'ner']).sents:
            sents.extend([_sent.strip() for _sent in sent.text.split('\n\n') if _sent.strip() != ''])
        return sents
    return sentencizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__file__)

    tokenizer, model, args = init()
    if args.coqa_file:
        try:
            logger.info('create sentencizer with spacy ...')
            sentencizer = create_sentencizer(spacy_model=args.spacy_model)
        except IOError as e:
            logger.warning('could not load spacy model "%s" for context sentence splitting. Please provide a list of strings as input for context.' % args.spacy_model)
            sentencizer = None
    if args.start_endpoint:
        try:
            logger.info('create wikipedia context fetcher ...')
            context_fetcher = create_wikipedia_context_fetcher(wikipedia_file=args.wikipedia_dump)
        except IOError as e:
            logger.warning('could not create a context fetcher. Please provide a context with every request.' % args.spacy_model)
            context_fetcher = None

    run(tokenizer, model, args)

