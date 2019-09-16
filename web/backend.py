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

import eventlet
import requests
import torch
from eventlet import wsgi
from flask import Flask, jsonify, Response, request, render_template

# Add the parent folder path to the sys.path list
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from interact import get_args, load_model, sample_sequence, norm_expl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)
#endpoint = Flask(__name__, static_url_path='')
app = Flask(__name__)
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


@app.errorhandler(InvalidUsage)
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


@app.route("/hello_world")
def hello_world():
    return "Hello World!"


@app.route("/", methods=['GET', 'POST'])
def _ask():
    return ask()


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


def insert_annotations(s, annotations, offset=0, exclude_key='text'):
    s_off = 0
    res = s
    for annot_key, annot in sorted(annotations.items(), key=lambda a: a[1]['offsetStart']):
        a_start = annot['offsetStart'] - offset
        a_len = annot['offsetEnd'] - annot['offsetStart']
        if 0 <= a_start < len(s) and 0 <= (a_start + a_len) < len(s):
            insert_opening = '<span class="named-entity" %s>' \
                             % " ".join(["data-" + k
                                         + "=\"" + (' '.join(v) if isinstance(v, list) else str(v)) + "\""
                                         for k, v in annot.items() if k != exclude_key])
            res = res[:a_start + s_off] + insert_opening + res[a_start + s_off:]
            s_off += len(insert_opening)
            insert_closing = '</span>'
            res = res[:a_start + a_len + s_off] + insert_closing + res[a_start + a_len + s_off:]
            s_off += len(insert_closing)

    return res


@app.route("/ask", methods=['GET', 'POST'])
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
        #if 'user_input' in params:
        #    raise DeprecationWarning('Parameter "user_input" is deprecated. Append the user_input to the history '
        #                             'instead.')

        # create required format of context: dict with entry_id -> list of sentences (strings)
        #if isinstance(params.get('background', None), str):
        #    params['background'] = {'user': params['background']}
        background = params.get('background', None) if params.get('keep_background', False) else None
        if isinstance(background, str):
            if background == '':
                background = None
            else:
                backgr_str = html.unescape(background)
                background = json.loads(backgr_str)
        if background is not None:
            for k in background:
                background[k]['external'] = True
        if not params.get('dont_fetch', False):
            assert context_fetcher is not None, 'No context/background fetcher initialized. Please provide a background with every request.'
            try:
                # use only the considered history to query background
                background = context_fetcher(' '.join(history[-(2 * args.max_history + 1):]), previous_context=background)
            except InvalidUsage as e:
                response = e.to_dict()
                response['status_code'] = e.status_code
                logger.warning(f'context_fetcher exception: {response}')
            except AssertionError as e:
                logger.warning(e)

        background_encoded = None
        if background is not None and len(background) > 0:
            background_keys, background_encoded = zip(*[(k, tokenizer.encode(b['text'])) for k, b in background.items()])
            params['background'] = background
        else:
            background_keys = []
            if 'background' in params:
                # delete for html template
                del params['background']

        personality_encoded = None
        if 'personality' in params:
            personality_encoded = tokenizer.encode(params['personality'])

        history_encoded = [tokenizer.encode(utterance) for utterance in history[-(2 * args.max_history + 1):]]
        utterance_types = params.get('utterance_types', None)

        if utterance_types is not None:
            assert len(history) == len(utterance_types), f'number of history elements [{len(history)}] does not match ' \
                                                       f'number of utterance_types [{len(utterance_types)}]'
            utterance_types_encoded = []
            allowed_hist_types = ', '.join(tokenizer.special_tokens.keys())
            for hist_type in utterance_types[-(2 * args.max_history + 1):]:
                assert hist_type in tokenizer.special_tokens, f'Unknown type for history element: {hist_type}. ' \
                                                              f'Use only these types: {allowed_hist_types}'
                utterance_types_encoded.append(tokenizer.special_tokens[hist_type])
        else:
            utterance_types_encoded = None
        # predict only if any history / user_input (was added to history) is available
        if len(history) > 0:
            # if explanations are requested:
            if params.get('explain', False):
                out_ids, eos, last_ids, explanations = sample_sequence(background=background_encoded,
                                                                       personality=personality_encoded,
                                                                       history=history_encoded,
                                                                       utterance_types=utterance_types_encoded,
                                                                       tokenizer=tokenizer,
                                                                       model=model, args=args,
                                                                       explain=True,
                                                                       replace_unknown=True)
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
                                                   history=history_encoded,
                                                   utterance_types=utterance_types_encoded,
                                                   tokenizer=tokenizer,
                                                   model=model, args=args,
                                                   explain=False,
                                                   replace_unknown=True)

            history_encoded.append(out_ids)
            params['prediction'] = tokenizer.decode(out_ids, skip_special_tokens=True)
            params['history'] = [tokenizer.decode(utterance) for utterance in history_encoded]
            params['eos'] = tokenizer.convert_ids_to_tokens([eos])[0]
            logger.debug('predicted:\n%s' % params['prediction'])

            # add annotations only when not explaining
            if not params.get('explain', False) and params.get('background', None) is not None:
                pos_start = 0
                params['history_annotated'] = []
                for h in params['history']:
                    params['history_annotated'].append(insert_annotations(s=h, annotations=params['background'], offset=pos_start))
                    # increase (1 for space)
                    pos_start += 1 + len(h)

        http_accept = params.get('HTTP_ACCEPT', False) or 'text/html'
        if 'application/json' in http_accept:
            json_data = json.dumps(params)
            response = Response(json_data, mimetype=http_accept)
        else:
            response = Response(render_template('chat.html', root=args.root, **params), mimetype='text/html')

        logger.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        raise InvalidUsage('%s: %s @line %s in %s' % (type(e).__name__, str(e), exc_tb.tb_lineno, fname))
    return response


@app.route("/reload", methods=['GET', 'POST'])
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
        if response.status_code != 200:
            raise InvalidUsage(message=response.text, status_code=response.status_code, payload=files)
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
                    res[wikipedia_entity_uri] = entity
                    logger.info('fetch entity data for "%s"...' % entity['rawName'])
                    wikipedia_id = entity['wikipediaExternalRef']
                    if wikipedia_data is not None:
                        # NOTE: keys in wikipedia_data may be strings!
                        wikipedia_entry = wikipedia_data.get(wikipedia_id, wikipedia_data.get(str(wikipedia_id), None))
                        if wikipedia_entry is not None:
                            logger.info('found entry for cuid=%i in wikipedia dump' % wikipedia_id)
                            if isinstance(wikipedia_entry['text'], list):
                                res[wikipedia_entity_uri]['text'] = ' '.join([s.strip() for s in wikipedia_entry['text']])
                            elif isinstance(wikipedia_entry['text'], str):
                                res[wikipedia_entity_uri]['text'] = wikipedia_entry['text']
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
                        res[wikipedia_entity_uri]['text'] = ' '.join(res_current_entity)
                else:
                    # overwrite all except 'text'
                    res[wikipedia_entity_uri].update(entity)

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
    if args.deploy:
        logger.info('use deployment server')
        wsgi.server(eventlet.listen(('', args.port)), app)
    else:
        app.run(host='0.0.0.0', port=args.port, debug=True)
