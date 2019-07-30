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
from collections import defaultdict

import requests
import torch
from flask import Flask, jsonify, Response, request, render_template

# Add the parent folder path to the sys.path list
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from interact import get_args, load_model, sample_sequence, norm_expl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)
#endpoint = Flask(__name__, static_url_path='')
endpoint = Flask(__name__)
#cors = CORS(endpoint)


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


def token_to_html(token, color):
    return '<span class="context" style="background-color:rgb(265, %i, %i)">%s</span>' % (color, color, html.escape(token))


def token_to_html_not_working(token, attribution):
    return f'<span class="context" attribution="{attribution}">{html.escape(token)}</span>'


def visualize_explanation(tokens, expl, split_tokens=(), return_tuples=False):
    expl = norm_expl(expl, _min=0.0)
    expl *= 256

    html_res = []
    current_html_res = ''
    prev_special = None
    for i in range(len(expl)):
        if tokens[i] in split_tokens:
            if len(current_html_res) > 0:
                if return_tuples:
                    html_res.append((prev_special, current_html_res))
                else:
                    html_res.append(prev_special + current_html_res)
                current_html_res = ''
            prev_special = tokens[i] if return_tuples else token_to_html(token=tokens[i], color=256 - int(expl[i]))
        else:
            current_html_res += token_to_html(token=tokens[i], color=256 - int(expl[i]))

    if return_tuples:
        html_res.append((prev_special, current_html_res))
    else:
        # allow not split result (no special_tokens given)
        html_res.append((prev_special or '') + current_html_res)
    return html_res


def process_explanations(explanations, last_ids, tokenizer):
    all_tokens = [tokenizer.decode([tok]) for tok in last_ids[0]]
    all_types = [tokenizer.decode([tok]) for tok in last_ids[1]]
    explanations_all_html = []
    explanations_html = {}
    explanations_sum = None
    for current_explanations in explanations:
        if explanations_sum is None:
            explanations_sum = current_explanations
        else:
            explanations_sum = {expl_type: explanations_sum[expl_type] + norm_expl(expl[:len(explanations_sum[expl_type])], _min=0.0) for expl_type, expl in current_explanations.items()}
        for expl_type, expl in current_explanations.items():
            explanations_html.setdefault(expl_type,[]).append('<div>%s</div>' % ''.join(visualize_explanation(tokens=all_tokens, expl=expl)))
        explanations_all_html.append('<div>%s</div>' % ''.join(visualize_explanation(tokens=all_tokens, expl=sum(current_explanations.values()))))

    # individual explanations per type and per prediction
    for expl_type, explanations_html in explanations_html.items():
        expl_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n%s</body>\n</html>' \
                    % '\n'.join(explanations_html)
        open('explanations_%s.html' % expl_type, 'w').write(expl_html)

    # individual explanations per prediction (summed over types)
    expl_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n%s</body>\n</html>' \
                       % '\n'.join(explanations_all_html)
    open('explanations.html', 'w').write(expl_html)

    # individual explanations per type (summed over predictions)
    for expl_type, expl in explanations_sum.items():
        expl_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n<div>%s</div>\n</body>\n</html>' \
                    % '\n'.join(['<div>%s</div>' % u for u in
                                 visualize_explanation(tokens=all_tokens, expl=explanations_sum[expl_type],
                                                       split_tokens=tokenizer.special_tokens.keys())])
        open('explanations_sum_%s.html' % expl_type, 'w').write(expl_html)

    # all summed together
    res = visualize_explanation(tokens=all_tokens, expl=sum(explanations_sum.values()), split_tokens=tokenizer.special_tokens.keys(), return_tuples=True)
    expl_html = '<!DOCTYPE html>\n<html>\n<head>\n<title>explanations</title>\n</head>\n<body>\n%s\n</body>\n</html>' \
                       % '\n'.join([f'<div>{html.escape(special_token)}:{u}</div>' for special_token, u in res])
    open('explanations_sum.html', 'w').write(expl_html)

    return res


@endpoint.route("/ask", methods=['GET', 'POST'])
def ask():
    try:
        start = time.time()
        logging.info('prediction requested')
        params = get_params()
        logger.debug(json.dumps(params, indent=2))
        history = params.get('history', '')
        if history == '':
            history = []
        elif isinstance(history, str):
            hist_str = html.unescape(history)
            history = json.loads(hist_str)
        user_input = params.get('user_input', None)
        if user_input is not None:
            history.append(user_input)

        # create required format of context: dict with entry_id -> list of sentences (strings)
        #if isinstance(params.get('background', None), str):
        #    params['background'] = {'user': params['background']}
        background = params.get('background', None)
        if isinstance(background, str):
            if background == '':
                background = None
            else:
                backgr_str = html.unescape(background)
                background = json.loads(backgr_str)
        if not params.get('dont_fetch', False):
            assert context_fetcher is not None, 'No context/background fetcher initialized. Please provide a background with every request.'
            try:
                background = context_fetcher(' '.join(history), previous_context=background)
            except AssertionError as e:
                logger.warning(e)
                pass

        background_encoded = None
        if background is not None and len(background) > 0:
            background_keys, background_encoded = zip(*[(k, tokenizer.encode(b)) for k, b in background.items()])
            params['background'] = background
        else:
            background_keys = []

        personality_encoded = None
        if 'personality' in params:
            personality_encoded = tokenizer.encode(params['personality'])

        history_encoded = [tokenizer.encode(utterance) for utterance in history[-(2 * args.max_history + 1):]]
        # predict only if any history / user_input (was added to history) is available
        if len(history) > 0:
            if params.get('explain', False):
                out_ids, eos, last_ids, explanations = sample_sequence(background=background_encoded, personality=personality_encoded,
                                                            history=history_encoded, tokenizer=tokenizer, model=model,
                                                            args=args, explain=params.get('explain', False))
                explanations_list = process_explanations(explanations=explanations, last_ids=last_ids, tokenizer=tokenizer)
                # add prediction
                explanations_list[-1] = (explanations_list[-1][0],
                                         f'<span class="prediction">{tokenizer.decode(out_ids)}</span>')
                params['explanation'] = {'history': [], 'background': {}}
                n_background = 0
                for special_token, expl_html in explanations_list:
                    if special_token in ['<user>', '<bot>']:
                        params['explanation']['history'].append(expl_html)
                    elif special_token == '<background>':
                        params['explanation']['background'][background_keys[n_background]] = expl_html
                        n_background += 1
            else:
                with torch.no_grad():
                    out_ids, eos = sample_sequence(background=background_encoded, personality=personality_encoded,
                                                   history=history_encoded, #[-(2 * args.max_history + 1):],
                                                   tokenizer=tokenizer, model=model, args=args,
                                                   explain=params.get('explain', False))

            history_encoded.append(out_ids)
            params['prediction'] = tokenizer.decode(out_ids, skip_special_tokens=True)
            params['history'] = [tokenizer.decode(utterance) for utterance in history_encoded]
            params['eos'] = tokenizer.convert_ids_to_tokens([eos])[0]
            logger.debug('predicted:\n%s' % params['prediction'])

        http_accept = params.get('HTTP_ACCEPT', False) or 'text/html'
        if 'application/json' in http_accept:
            json_data = json.dumps(params)
            response = Response(json_data, mimetype=http_accept)
        else:
            response = Response(render_template('chat.html', **params), mimetype='text/html')

        logger.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        raise InvalidUsage('%s: %s @line %s in %s' % (type(e).__name__, str(e), exc_tb.tb_lineno, fname))
    return response


@endpoint.route("/reload", methods=['GET', 'POST'])
def reload_model():
    global model, tokenizer
    try:
        start = time.time()
        logging.info('prediction requested')
        params = get_params()
        logger.debug(json.dumps(params, indent=2))
        args_model_checkpoint = args.model_checkpoint
        full_model_checkpoint = os.path.join(os.path.dirname(args_model_checkpoint), params['model_checkpoint'])
        model, tokenizer = load_model(model_checkpoint=full_model_checkpoint, model_type=params['model_type'])

        response = Response(f'loaded model {params["model_checkpoint"]} successfully')
        logger.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        raise InvalidUsage('%s: %s @line %s in %s' % (type(e).__name__, str(e), exc_tb.tb_lineno, fname))
    return response


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
        if len(s) == 0:
            logger.warning('input for context_fetcher is an empty string')
            return res
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

    args = get_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, tokenizer = load_model(model_checkpoint=args.model_checkpoint, model_type=args.model)

    try:
        logger.info('create wikipedia context fetcher ...')
        context_fetcher = create_wikipedia_context_fetcher(wikipedia_file=args.wikipedia_dump)
    except IOError as e:
        logger.warning(
            'could not create a context fetcher. Please provide a context with every request.' % args.spacy_model)
        context_fetcher = None

    logger.info('Starting the API')
    # endpoint.static = 'static'
    endpoint.run(host='0.0.0.0', port=5000)#, debug=True)
