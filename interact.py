# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ast
import html
import json
import logging
import os
import pickle
import random
import re
import sys
import time
import traceback
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat

import numpy as np
import requests
from tqdm import tqdm
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, Response, request

from train import MODELS, build_input_from_segments, TYPE_BACKGROUND, TYPE_BOT, TYPE_USER, TYPE_BOT_DEPRECATED, \
    TYPE_USER_DEPRECATED
from utils import get_dataset_personalities, download_pretrained_model#, create_wikipedia_context_fetcher

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


def sample_sequence(tokenizer, model, args, background=None, personality=None, history=(), current_output=None, explain=False):
    max_sequence_length = args.max_sequence_length if args.max_sequence_length > 0 else model.config.n_ctx
    assert max_sequence_length <= model.config.n_ctx, 'max_sequence_length [%i] was set to a value higher than ' \
                                                      'supported by the model (config.n_ctx [%i]). Please use a lower ' \
                                                      'value or do not set it [-1] to use the highest supported one.' \
                                                      % (max_sequence_length, model.config.n_ctx)
    special_tokens_ids = tokenizer.special_tokens.values()
    # causes strange behaviour:
    #type_bot = tokenizer.special_tokens.get(TYPE_BOT, tokenizer.special_tokens[TYPE_BOT_DEPRECATED])
    #type_user = tokenizer.special_tokens.get(TYPE_USER, tokenizer.special_tokens[TYPE_USER_DEPRECATED])
    type_bot = tokenizer.special_tokens[TYPE_BOT]
    type_user = tokenizer.special_tokens[TYPE_USER]
    # default to speaker2 if background is not present in model
    type_background = tokenizer.special_tokens.get(TYPE_BACKGROUND, type_user)
    #logger.debug('expected sequence length (without prediction): %i; max_allowed: %i (inclusive prediction)'
    #             % (len(list(chain(*(context + history)))) + len(history) + 1, max_sequence_length))
    context = []
    if background is not None:
        if isinstance(background, list):
            context.extend([(type_background, b) for b in background])
        else:
            context.append((type_background, background))
    if personality is not None:
        context.append((type_bot, personality))
    if current_output is None:
        current_output = []
    _history = [(type_user if (len(history) - i) % 2 else type_bot, h) for i, h in enumerate(history)]
    eos = None
    explanations = []
    last_ids = None
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(context=context,
                                                       history=_history,
                                                       reply=(type_bot, current_output), tokenizer=tokenizer, eos=None,
                                                       max_sequence_length=max_sequence_length)
        l_trunc = len(list(chain(*sequence))) - len(instance['input_ids'])
        assert l_trunc <= 0, 'The sequence was truncated. Please provide less context + history + question!'

        if torch.is_grad_enabled():
            model.zero_grad()

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits_all = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits_all.clone(), top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        #logger.debug('nbr of non zeros in filtered probs_top: %i (of %i)' % (torch.nonzero(probs.data).size(0), len(probs)))

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                logger.debug('resample because of special token...')
                prev = torch.multinomial(probs, num_samples=1)


        if explain:
            probs_all = F.softmax(logits_all, dim=-1)
            #logger.debug('nbr of non zeros in filtered probs_all: %i (of %i)'
            #             % (torch.nonzero(probs_all.data).size(0), len(probs)))
            #logger.debug('probs_all min: %f, max: %f; logits_all min: %f, max %f'
            #             % (torch.min(probs_all).item(), torch.max(probs_all).item(),
            #                torch.min(probs_all).item(), torch.max(probs_all).item()))
            #logger.debug('probs_top min: %f, max: %f; logits_top min: %f, max %f'
            #             % (torch.min(probs).item(), torch.max(probs).item(),
            #                torch.min(logits).item(), torch.max(logits).item()))
            prev_prob = probs_all[prev]
            logger.debug('prob for current sample [%s]: %f' % (tokenizer.decode([prev.item()]), prev_prob.item()))
            prev_prob.backward()
            model_wte = model.transformer.wte.weight
            if torch.min(model_wte.grad) == torch.max(model_wte.grad) == 0.0:
                logger.warning('create explanations (i: %i): min==max==0.0 for ALL embedding gradients' % len(current_output))
            grads_input_ids = model_wte.grad[input_ids.squeeze()]
            grads_token_type_ids = model_wte.grad[token_type_ids.squeeze()]
            if torch.min(grads_input_ids) == torch.max(grads_input_ids) == 0.0:
                logger.warning('create explanations (i: %i): min==max==0.0 for gradients wrt. input_ids' % len(current_output))
            if torch.min(grads_token_type_ids) == torch.max(grads_token_type_ids) == 0.0:
                logger.warning('create explanations (i: %i): min==max==0.0 for gradients wrt. grads_token_type_ids' % len(current_output))
            expl_input_ids = (torch.abs(grads_input_ids) * torch.abs(model_wte[input_ids.squeeze()])).sum(dim=-1)
            expl_token_type_ids = (torch.abs(grads_token_type_ids) * torch.abs(model_wte[token_type_ids.squeeze()])).sum(dim=-1)
            last_ids = (instance["input_ids"], instance["token_type_ids"])
            explanations.append((expl_input_ids.detach().cpu().numpy(), expl_token_type_ids.cpu().detach().numpy()))

        if prev.item() in special_tokens_ids:
            eos = prev.item()
            break
        current_output.append(prev.item())

    if explain:
        return current_output, eos, last_ids, explanations

    return current_output, eos

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


def visualize_explanation(tokens, expl, special_tokens=()):
    expl = norm_expl(expl, _min=0.0)
    expl *= 256

    html_res = []
    current_html_res = ''
    for i in range(len(expl)):
        if tokens[i] in special_tokens and len(current_html_res) > 0:
            html_res.append(current_html_res)
            current_html_res = ''
        c = 256 - int(expl[i])
        current_html_res += '<span style="background-color:rgb(265, %i, %i)">%s</span>' % (c, c, html.escape(tokens[i]))
    html_res.append(current_html_res)
    return html_res

def norm_expl(expl, _min=None, square=False):
    if square:
        expl = expl * expl
    if _min is None:
        _min = np.min(expl)
    _max = np.max(expl)
    if _max != _min:
        expl /= _max - _min
    else:
        logger.warning('explanation max==min==%f' % _max)
    return expl

def process_explanations(explanations, last_ids, tokenizer):
    all_tokens = [tokenizer.decode([tok]) for tok in last_ids[0]]
    all_types = [tokenizer.decode([tok]) for tok in last_ids[1]]
    token_explanations_html = []
    type_explanations_html = []
    explanations_html = []
    token_explanation_sum = None
    type_explanation_sum = None
    for token_explanation, type_explanation in explanations:
        if token_explanation_sum is None:
            token_explanation_sum = token_explanation
        else:
            token_explanation_sum = token_explanation_sum + norm_expl(token_explanation[:len(token_explanation_sum)], _min=0.0)
        if type_explanation_sum is None:
            type_explanation_sum = type_explanation
        else:
            type_explanation_sum = type_explanation_sum + norm_expl(type_explanation[:len(token_explanation_sum)], _min=0.0)
        explanations_html.append(
            '<div>%s</div>' % ''.join(visualize_explanation(tokens=all_tokens, expl=token_explanation + type_explanation)))
        token_explanations_html.append(
            '<div>%s</div>' % ''.join(visualize_explanation(tokens=all_tokens, expl=token_explanation)))
        type_explanations_html.append('<div>%s</div>' % ''.join(visualize_explanation(tokens=all_tokens, expl=type_explanation)))

    explanation_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n%s</body>\n</html>' \
                       % '\n'.join(explanations_html)
    open('explanations.html', 'w').write(explanation_html)
    token_explanation_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n%s</body>\n</html>' \
                             % '\n'.join(token_explanations_html)
    open('explanations_token.html', 'w').write(token_explanation_html)
    type_explanation_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n%s</body>\n</html>' \
                            % '\n'.join(type_explanations_html)
    open('explanations_type.html', 'w').write(type_explanation_html)

    res = visualize_explanation(tokens=all_tokens, expl=token_explanation_sum + type_explanation_sum, special_tokens=tokenizer.special_tokens.keys())
    explanation_sum_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n%s\n</body>\n</html>' \
                       % '\n'.join(['<div>%s</div>' % u for u in res])
    open('explanations_sum.html', 'w').write(explanation_sum_html)
    token_explanation_sum_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n<div>%s</div>\n</body>\n</html>' \
                                 % '\n'.join(['<div>%s</div>' % u for u in visualize_explanation(tokens=all_tokens, expl=token_explanation_sum, special_tokens=tokenizer.special_tokens.keys())])
    open('explanations_sum_token.html', 'w').write(token_explanation_sum_html)
    type_explanation_sum_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n<div>%s</div>\n</body>\n</html>' \
                                % '\n'.join(
        ['<div>%s</div>' % u for u in visualize_explanation(tokens=all_tokens, expl=type_explanation_sum, special_tokens=tokenizer.special_tokens.keys())])
    open('explanations_sum_type.html', 'w').write(type_explanation_sum_html)
    return res


@endpoint.route("/ask", methods=['GET', 'POST'])
def ask():
    try:
        start = time.time()
        logging.info('prediction requested')
        params = get_params()
        logger.debug(json.dumps(params, indent=2))
        history = params.get('history', [])
        user_input = params['user_input']
        history.append(user_input)

        # create required format of context: dict with entry_id -> list of sentences (strings)
        if isinstance(params.get('background', None), str):
            params['background'] = {'user': params['background']}
        background = params.get('background', None)
        if not params.get('dont_fetch', False):
            assert context_fetcher is not None, 'No context/background fetcher initialized. Please provide a background with every request.'
            try:
                background = context_fetcher(' '.join(history), previous_context=background)
            except AssertionError as e:
                logger.warning(e)
                pass

        background_encoded = None
        if background is not None:
            background_encoded = [tokenizer.encode(b) for b in background.values()]
            params['background'] = background

        personality_encoded = None
        if 'personality' in params:
            personality_encoded = tokenizer.encode(params['personality'])

        history_encoded = [tokenizer.encode(utterance) for utterance in history]

        if params.get('explain', False):

            out_ids, eos, last_ids, explanations = sample_sequence(background=background_encoded, personality=personality_encoded,
                                                        history=history_encoded, tokenizer=tokenizer, model=model,
                                                        args=args, explain=params.get('explain', False))
            params['explanation'] = process_explanations(explanations=explanations, last_ids=last_ids, tokenizer=tokenizer)
            params['explanation'][-1]+= '<span style="background-color:grey">%s</span>' \
                                        % tokenizer.decode(out_ids, skip_special_tokens=False)

            resp_html = '\n'.join(['<div>%s</div>' % u for u in params['explanation']])
            resp_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explained response</title>\n</head>\n<body>\n%s</body>\n</html>' % resp_html
            return_type = 'text/html'
            response = Response(resp_html, mimetype=return_type)
            logger.info("Time spent handling the request: %f" % (time.time() - start))
        else:
            with torch.no_grad():
                out_ids, eos = sample_sequence(background=background_encoded, personality=personality_encoded,
                                               history=history_encoded, tokenizer=tokenizer, model=model, args=args,
                                               explain=params.get('explain', False))

            history_encoded.append(out_ids)
            history_encoded = history_encoded[-(2 * args.max_history + 1):]
            params['prediction'] = tokenizer.decode(out_ids, skip_special_tokens=True)
            params['history'] = [tokenizer.decode(utterance) for utterance in history_encoded]
            params['eos'] = tokenizer.convert_ids_to_tokens([eos])[0]
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
            background_sents = sentencizer(instance['story'])
            background_encoded = tokenizer.encode(' '.join([sentence.strip() for sentence in background_sents]))
            history_encoded = []
            for question in instance['questions']:
                n_total += 1
                question_text = question['input_text']
                history_encoded.append(tokenizer.encode(question_text))
                with torch.no_grad():
                    try:
                        out_ids, _ = sample_sequence(background=background_encoded, history=history_encoded, tokenizer=tokenizer,
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
        personality = chain(*random.choice(personalities))
        history_encoded = []
        logger.info("Selected personality: %s", tokenizer.decode(personality))
        while True:
            raw_text = input(">>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input(">>> ")
            history_encoded.append(tokenizer.encode(raw_text))
            with torch.no_grad():
                out_ids, _ = sample_sequence(personality=personality, history=history_encoded, tokenizer=tokenizer,
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


def create_wikipedia_context_fetcher(wikipedia_file=None):
    url_disambiguate = "http://cloud.science-miner.com/nerd/service/disambiguate"
    wikipedia_data = None
    if wikipedia_file:
        logger.info('load wikipedia data from: %s...' % wikipedia_file)
        try:
            # expects a pickel file containing a dict: wikipedia cuid -> {'text': [['This is a sentence.'], ['This is another sentence.']]}
            wikipedia_data = pickle.load(open(wikipedia_file, "rb"))
            logger.info('loaded %i articles' % len(wikipedia_data))
        except:
            logger.error('could not load wikipedia dump: %s' % wikipedia_file)
    url_fetch = "http://cloud.science-miner.com/nerd/service/kb/concept"
    headers = {
        'Cache-Control': 'no-cache',
    }

    wikipedia_base_uri = "https://en.wikipedia.org/wiki?curid="


    def _context_fetcher(s, previous_context=None):
        logger.info('fetch context for "%s"...' % s)
        res = previous_context or {}
        query = {'text': s, "language": {"lang": "en"}}
        files = {'query': (None, json.dumps(query))}
        response = requests.post(url_disambiguate, headers=headers, files=files, timeout=60)
        response_data = json.loads(response.text)
        if len(response_data.get('entities', [])) > 0:
            for entity in response_data['entities']:
                if 'wikipediaExternalRef' not in entity:
                    logger.warning('entity "%s" (selection confidence: %s; disambiguation confidence: %s) has no wikipediaExternalRef. Skip it.'
                                   % (entity['rawName'], entity['nerd_selection_score'], entity['nerd_score']))
                    continue
                wikipedia_entity_uri = wikipedia_base_uri + str(entity['wikipediaExternalRef'])
                logger.debug('found "%s" (%s) with selection confidence of %s and disambiguation confidence of %s'
                             % (entity['rawName'], wikipedia_entity_uri, entity['nerd_selection_score'], entity['nerd_score']))
                # fetch only not already available context
                if wikipedia_entity_uri not in res:
                    logger.info('fetch entity data for "%s"...' % entity['rawName'])
                    wikipedia_id = entity['wikipediaExternalRef']
                    if wikipedia_data is not None:
                        # NOTE: keys in wikipedia_data may be strings!
                        wikipedia_entry = wikipedia_data.get(wikipedia_id, wikipedia_data.get(str(wikipedia_id), None))
                        if wikipedia_entry is not None:
                            logger.info('found entry for cuid=%i in wikipedia dump' % wikipedia_id)
                            if isinstance(wikipedia_entry['text'], list):
                                res[wikipedia_entity_uri] = ' '.join([s.strip() for s in wikipedia_entry['text']])
                            elif isinstance(wikipedia_entry['text'], str):
                                res[wikipedia_entity_uri] = wikipedia_entry['text']
                            else:
                                raise NotImplementedError('wikidata entry for "%s" has unknown format (only list or str are allowed): %s'
                                                          % (wikipedia_entity_uri, str(wikipedia_entry['text'])))
                            continue
                        else:
                            logger.warning('cuid=%i not found in wikipedia dump. Fetch data from %s...' % (wikipedia_id, url_fetch))
                    response_entity = requests.get('%s/%s?lang=en' % (url_fetch, wikipedia_id), timeout=120)
                    if response_entity.status_code != requests.codes.ok:
                        logger.warning('Failed to fetch data for "%s" (url: %s, http status code: %s). Skip it.'
                                       % (wikipedia_entity_uri, response_entity.request.url, response_entity.status_code))
                        continue
                    response_entity_data = json.loads(response_entity.text)
                    #res[wikipedia_entity_uri] = []
                    res_current_entity = []
                    #assert len(response_entity_data['definitions']) > 0, 'no definitions found for entity: %s' % entity['rawName']
                    for definition in response_entity_data['definitions']:
                        if definition.get('lang', '') == 'en':
                            definition_cleaned = definition['definition']
                            # remove links, e.g. "[[Western civilisation]]" or "the [[Diocese of Rome|Bishop of Rome]]"
                            definition_cleaned = re.sub(r"\[\[(?:[^\]]*?\|)?([^\]]*?)\]\]", r"\1", definition_cleaned)
                            definition_cleaned = definition_cleaned.replace("'''", '"')
                            res_current_entity.append(definition_cleaned.strip())
                    if len(res_current_entity) > 0:
                        res[wikipedia_entity_uri] = ' '.join(res_current_entity)

        assert len(res) > 0, 'no context found (entities found: %s)' % str([entity['rawName'] for entity in response_data.get('entities', [])])
        return res
    return _context_fetcher


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

