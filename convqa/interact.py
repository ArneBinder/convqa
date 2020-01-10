# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import random
from argparse import ArgumentParser
from contextlib import contextmanager
from itertools import chain
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train import MODELS, build_input_from_segments, TYPE_BACKGROUND, TYPE_BOT, TYPE_USER
from utils import get_dataset_personalities, download_pretrained_model, create_sentencizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)


@contextmanager
def capture_inputs_with_gradients(module: torch.nn.modules.module):
    """
    Inspired by https://github.com/allenai/allennlp/blob/417a75727d4604e1da0eda461b03b29a73f0bf50/allennlp/predictors/predictor.py#L146-L178

    Context manager that captures the inputs and gradients at a certain module.
    The idea is that you could use it as follows:
    .. code-block:: python
        with capture_inputs_with_gradients(module) as captured:
            outputs = predictor.predict_json(inputs)
            selected_scalar_output = ...
            selected_scalar_output.backward()

        # use captured['inputs'] and captured['gradients] here!
    """
    if module is not None:
        # dict is required to keep a reference
        results = {}

        # First we'll register hooks to add the outputs of each module to the results dict.
        def capture_input():
            def _capture_input(mod, inputs, _):
                results['inputs'] = inputs

            return _capture_input

        fw_hook = module.register_forward_hook(capture_input())

        def capture_gradients():
            def _capture_grads(mod, grad_input, grad_output):
                results['grads'] = grad_input

            return _capture_grads

        bw_hook = module.register_backward_hook(capture_gradients())

        # If you capture the return value of the context manager, you get the results dict.
        yield results

        # And then when you exit the context we remove all the hooks.
        fw_hook.remove()
        bw_hook.remove()
    else:
        yield None


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


def get_embedding_explanations(weights, ids, label='embedding'):
    if torch.min(weights.grad) == torch.max(weights.grad) == 0.0:
        logger.warning(f'create explanations (i: {len(ids)}): min==max==0.0 for ALL {label} gradients')
    grads_ids = weights.grad[ids.squeeze()]
    if torch.min(grads_ids) == torch.max(grads_ids) == 0.0:
        logger.warning(f'create explanations (i: {len(ids)}): min==max==0.0 for {label} gradients wrt. input_ids')
    expl_ids = (torch.abs(grads_ids) * torch.abs(weights[ids])).sum(dim=-1)
    return expl_ids


# TODO: try sampling like in https://einstein.ai/presentations/ctrl.pdf
def sample_sequence(tokenizer, model, args, background=None, personality=None, utterances=(), utterance_types=(),
                    current_output=None, explain=False, replace_unknown=False):
    max_sequence_length = args.max_sequence_length if args.max_sequence_length > 0 else model.config.n_ctx
    assert max_sequence_length <= model.config.n_ctx, 'max_sequence_length [%i] was set to a value higher than ' \
                                                      'supported by the model (config.n_ctx [%i]). Please use a lower ' \
                                                      'value or do not set it [-1] to use the highest supported one.' \
                                                      % (max_sequence_length, model.config.n_ctx)

    special_tokens_ids = tokenizer.all_special_ids
    bot_token_id = tokenizer.convert_tokens_to_ids(TYPE_BOT)
    user_token_id = tokenizer.convert_tokens_to_ids(TYPE_USER)
    #eos_token_id = tokenizer.eos_token_id
    # default to speaker2 if background is not present in model
    #background_token_id = tokenizer.special_tokens.get(TYPE_BACKGROUND, user_token_id)
    background_token_id = tokenizer.convert_tokens_to_ids(TYPE_BACKGROUND)
    #logger.debug('expected sequence length (without prediction): %i; max_allowed: %i (inclusive prediction)'
    #             % (len(list(chain(*(context + history)))) + len(history) + 1, max_sequence_length))
    context = []
    if background is not None:
        if isinstance(background, list) or isinstance(background, tuple):
            context.extend([(background_token_id, b) for b in background])
        else:
            context.append((background_token_id, background))
    if personality is not None:
        context.append((bot_token_id, personality))
    if current_output is None:
        current_output = []
    assert len(utterances) == len(utterance_types), f'length of utterances [{len(utterances)}] has to be the ' \
                                                    f'same as length of utterance_types [{len(utterance_types)}], ' \
                                                    f'if that is provided'
    utterances_with_types = list(zip(utterance_types, utterances))
    eos = None
    explanations = []
    last_ids = None
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(context=context,
                                                       history=utterances_with_types,
                                                       reply=(bot_token_id, current_output), tokenizer=tokenizer,
                                                       eos=None,
                                                       max_sequence_length=max_sequence_length)
        l_trunc = len(list(chain(*sequence))) - len(instance['input_ids'])
        assert l_trunc <= 0, 'The sequence was truncated. Please provide less context + history + question!'

        if torch.is_grad_enabled():
            model.zero_grad()

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        if explain:
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        else:
            position_ids = None

        capture_at_module = None
        if explain:
            if "gpt2" == args.model:
                capture_at_module = model.transformer.drop
            else:
                raise NotImplementedError('explain is implemented only for gpt2 model')

        # this captures only if capture_at_module != None
        with capture_inputs_with_gradients(capture_at_module) as captured:
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

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
                #logger.debug('prob for current sample [%s]: %f' % (tokenizer.decode([prev.item()]), prev_prob.item()))
                prev_prob.backward()

        if explain:
            assert captured is not None, \
                f'no captured inputs an gradients available to create explanations (generated token position: ' \
                f'{len(current_output) + 1})'
            expl_internal = (torch.abs(captured['grads'][0][0]) * torch.abs(captured['inputs'][0][0])).sum(dim=-1)

            last_ids = (instance["input_ids"], instance["token_type_ids"])
            explanations.append({'internal': expl_internal.detach().cpu().numpy()})

        if prev.item() in special_tokens_ids:
            eos = prev.item()
            break
        current_output.append(prev.item())

    if current_output == tokenizer.encode('unknown'):
        current_output = tokenizer.encode('i don\'t know')

    if explain:
        return current_output, eos, last_ids, explanations

    return current_output, eos


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


def load_model(model_checkpoint, model_type):
    if model_checkpoint == "":
        model_checkpoint = download_pretrained_model()
    else:
        assert os.path.exists(model_checkpoint), f'checkpoint directory not found: {model_checkpoint}'

    logger.info("Get pretrained model and tokenizer")
    if model_type not in MODELS:
        raise NotImplementedError('model "%s" not implemented. use one of %s' % (model_type, str(MODELS.keys)))
    config_class, tokenizer_class, _, model_class = MODELS[model_type]

    _tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    _model = model_class.from_pretrained(model_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)
    _model.eval()
    return _model, _tokenizer, os.path.basename(model_checkpoint) if model_checkpoint else model_checkpoint


def get_args(parser=ArgumentParser(), arguments=()):
    """
    :param parser: optional argument parser
    :param arguments: list/tuple of dicts to create additional parser arguments from (via parser.add_argument)
    :return: parsed arguments
    """
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
    parser.add_argument("--wikipedia_dump", type=str, default="", help="path to a pickle file containing a dict: "
                                                                       "wikipedia cuid -> {'text': ['This is a sentence.', 'This is another sentence.']}")
    # for interact mode
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    # for file processing
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="to allow automatic sentence splitting for flask endpoint")
    parser.add_argument("--coqa_file", type=str, default="", help="path to a file in the CoQA dataset format containing question where the answers will be predicted")
    parser.add_argument("--prediction_out", type=str, default="", help="path to a file to save the predictions")

    # api
    parser.add_argument("--port", type=int, default=5000, help="port of the started endpoint")
    parser.add_argument("--deploy", action="store_true", help="create a wsgi deployment server")
    parser.add_argument("--root", type=str, default='', help='if started behind a reverse proxy, '
                                                             'use e.g. "--root backend/"')
    parser.add_argument("--entity_linking_service_url", type=str, default='http://cloud.science-miner.com/nerd/service',
                        help="base url for the entity linking service (entity-fishing)")
    parser.add_argument("--ssl-dir", type=str, default='',
                        help="if not empty and in deploy mode, use <ssl-dir>/create ssl connection and")

    for parse_arg in arguments:
        parser.add_argument(**parse_arg)

    args = parser.parse_args()
    logger.info(pformat(args))

    return args


def process_coqa_file(tokenizer, model, args):
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
        history_types_encoded = []
        for question in instance['questions']:
            n_total += 1
            question_text = question['input_text']
            history_encoded.append(tokenizer.encode(question_text))
            history_types_encoded.append(tokenizer.convert_tokens_to_ids(TYPE_USER))
            with torch.no_grad():
                try:
                    out_ids, _ = sample_sequence(background=background_encoded, utterances=history_encoded,
                                                 utterance_types=history_types_encoded,
                                                 tokenizer=tokenizer,
                                                 model=model, args=args)
                    history_encoded.append(out_ids)
                    history_types_encoded.append(tokenizer.convert_tokens_to_ids(TYPE_BOT))
                    history_encoded = history_encoded[-(2 * args.max_history + 1):]
                    history_types_encoded = history_types_encoded[-(2 * args.max_history + 1):]
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


def run_interactive(tokenizer, model, args):
    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    personality = list(chain(*random.choice(personalities)))
    history_encoded = []
    history_types_encoded = []
    logger.info("Selected personality: %s", tokenizer.decode(personality))
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history_encoded.append(tokenizer.encode(raw_text))
        history_types_encoded.append(tokenizer.convert_tokens_to_ids(TYPE_USER))
        with torch.no_grad():
            out_ids, _ = sample_sequence(personality=personality, utterances=history_encoded,
                                         utterance_types=history_types_encoded, tokenizer=tokenizer,
                                         model=model, args=args)
        history_encoded.append(out_ids)
        history_types_encoded.append(tokenizer.convert_tokens_to_ids(TYPE_BOT))
        history_encoded = history_encoded[-(2 * args.max_history + 1):]
        history_types_encoded = history_types_encoded[-(2 * args.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)


if __name__ == "__main__":

    args = get_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, tokenizer, checkpoint_fn = load_model(model_checkpoint=args.model_checkpoint, model_type=args.model)

    if args.coqa_file:
        try:
            logger.info('create sentencizer with spacy ...')
            sentencizer = create_sentencizer(spacy_model=args.spacy_model)
        except IOError as e:
            logger.warning('could not load spacy model "%s" for context sentence splitting. Please provide a list of strings as input for context.' % args.spacy_model)
            sentencizer = None

    if args.coqa_file:
        process_coqa_file(tokenizer=tokenizer, model=model, args=args)
    else:
        run_interactive(tokenizer=tokenizer, model=model, args=args)
