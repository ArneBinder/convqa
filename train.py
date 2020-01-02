# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict, Counter
from itertools import chain

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import CONFIG_NAME, WEIGHTS_NAME, GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel, AdamW, \
    get_linear_schedule_with_warmup, GPT2Config

from utils import get_dataset


class GradReverse(Function):
    @staticmethod
    def forward(ctx, input):
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class GPT2ClassifierHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config, num_labels, invert_gradients=False):
        super(GPT2ClassifierHead, self).__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.num_labels = num_labels
        self.linear = nn.Linear(config.n_embd, self.num_labels)
        self.invert_gradients = invert_gradients


        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        # Classification logits
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
        # mc_token_ids (bsz, num_choices)
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        # (bsz, num_choices, 1, hidden_size)
        classifier_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        # invert gradients
        if self.invert_gradients:
            classifier_h = grad_reverse(classifier_h)
        # (bsz, num_choices, hidden_size)
        classifier_h = self.dropout(classifier_h.transpose(1, 2)).transpose(1, 2)
        classifier_logits = self.linear(classifier_h)
        # (bsz, num_choices, num_labels)
        return classifier_logits


class GPT2DoubleHeadsModelwithAdversarial(GPT2DoubleHeadsModel):
    """OpenAI GPT-2 model with a Language Modeling and a Multiple Choice head ("Language Models are Unsupervised Multitask Learners").

        Params:
            config: a GPT2Config class instance with the configuration to build a new model

        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
                indices selected in the range [0, config.vocab_size[
            `mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
                which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
            `position_ids`: an optional torch.LongTensor with the same shape as input_ids
                with the position indices (selected in the range [0, config.n_positions - 1[.
            `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
                You can use it to add a third type of embedding to each input token in the sequence
                (the previous two being the word and position embeddings).
                The input, position and token_type embeddings are summed inside the Transformer before the first
                self-attention block.
            `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
                with indices selected in [-1, 0, ..., config.vocab_size]. All labels set to -1 are ignored (masked), the loss
                is only computed for the labels set in [0, ..., config.vocab_size]
            `mc_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
                with indices selected in [0, ..., num_choices].
            `adv_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size, num_choices]
                with indices selected in [0, ..., num_adv_labels].

            `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
                (key and values in the attention blocks) to speed up sequential decoding
                (this is the presents output of the model, cf. below).

        Outputs:
            if `lm_labels` and `multiple_choice_labels` are not `None`:
                Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
            else: a tuple with
                `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, config.vocab_size]
                `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]
                `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                    torch.FloatTensors. They can be reused to speed up sequential decoding.

        Example usage:
        ```python
        # Already been converted into BPE token ids
        input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
        mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)

        config = modeling_gpt2.GPT2Config()

        model = modeling_gpt2.GPT2LMHeadModel(config)
        lm_logits, multiple_choice_logits, presents = model(input_ids, mc_token_ids)
        ```
        """
    def __init__(self, config, num_adv_labels, **kwargs):
        super(GPT2DoubleHeadsModelwithAdversarial, self).__init__(config, **kwargs)
        self.adv_classifier_head = GPT2ClassifierHead(config, num_labels=num_adv_labels, invert_gradients=True) if num_adv_labels > 1 else None
        self.num_adv_labels = num_adv_labels

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, adv_labels=None, token_type_ids=None,
                position_ids=None, past=None):
        transformer_output = self.transformer(input_ids, position_ids, token_type_ids, past)
        if self.transformer.output_attentions:
            all_attentions, hidden_states, presents = transformer_output
        else:
            hidden_states, presents = transformer_output
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        adv_logits = self.adv_classifier_head(hidden_states, mc_token_ids) if self.adv_classifier_head is not None else None
        losses = []
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if  adv_labels is not None and adv_logits is not None:
            loss_fct = CrossEntropyLoss()
            loss_adv = loss_fct(adv_logits.view(-1, self.adv_classifier_head.num_labels), adv_labels.view(-1))
            losses.append(loss_adv)
        if losses:
            return losses
        if self.transformer.output_attentions:
            return all_attentions, lm_logits, mc_logits, presents
        return lm_logits, mc_logits, presents


TYPE_BOS = "<bos>"
TYPE_EOS = "<eos>"
TYPE_PAD = "<pad>"
TYPE_BACKGROUND = "<background>"
TYPE_BOT = "<bot>"
TYPE_USER = "<user>"
TYPE_BOT_DEPRECATED = "<speaker1>"
TYPE_USER_DEPRECATED = "<speaker2>"
#SPECIAL_TOKENS = [TYPE_BOS, TYPE_EOS, TYPE_BACKGROUND, TYPE_BOT, TYPE_USER, TYPE_PAD]
#SPECIAL_TOKENS = {'background_token': TYPE_BACKGROUND, 'bot_token': TYPE_BOT, 'user_token': TYPE_USER}
#SPECIAL_TOKENS = [TYPE_BACKGROUND, TYPE_BOT, TYPE_USER]
#MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "adv_labels", "token_type_ids"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

# implemented models:
#   <model name> -> (<Tokenizer class>, <DoubleHeadsModel class>, <LMHeadModel>)
# see huggingface/pytorch-pretrained-bert for available model names
MODELS = {
    #'openai-gpt': (OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel),
    #'gpt2': (GPT2Tokenizer, GPT2DoubleHeadsModelwithAdversarial, GPT2LMHeadModel),
    'gpt2': (GPT2Config, GPT2Tokenizer, GPT2DoubleHeadsModel, GPT2LMHeadModel),
}

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    l_counter = Counter((len(x) for x in dataset["input_ids"]))
    max_l = max(l_counter.keys())

    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(context, history, reply, tokenizer, lm_labels=False, eos=None,
                              return_strings=False, max_sequence_length=None):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos = tokenizer.bos_token_id if not return_strings else tokenizer.bos_token

    instance = {}
    sequence = context + history + [reply]
    sequence = [[bos]] + [[speaker] + u for speaker, u in sequence]

    if eos is not None:
        sequence[-1].append(eos)

    instance["input_ids"] = list(chain(*sequence))
    # set types by first element of each sequence
    instance["token_type_ids"] = [s[0] for i, s in enumerate(sequence) for _ in s]
    # DEBUG:
    # list(zip(tokenizer.convert_ids_to_tokens(instance['token_type_ids']), tokenizer.convert_ids_to_tokens(instance['input_ids'])))
    # list(zip(tokenizer.convert_ids_to_tokens(instance['token_type_ids'][-30:]), tokenizer.convert_ids_to_tokens(instance['input_ids'][-30:])))

    if max_sequence_length:
        instance["input_ids"] = instance["input_ids"][:max_sequence_length]
        instance["token_type_ids"] = instance["token_type_ids"][:max_sequence_length]

    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        # predict next words only for last sequence (reply), but w/o to predict initial special token, e.g. <speaker1>!
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        if max_sequence_length:
            instance["lm_labels"] = instance["lm_labels"][:max_sequence_length]
    return instance, sequence


def create_typed_utterance(utt, default_type, allow_not_an_utterance=False):
    # an utterance is a [list of ints], i.e. not typed words, or [int, [list of ints]], i.e. typed list of words
    if isinstance(utt, list) and isinstance(utt[0], int):
        # typed utterance
        if len(utt) == 2 and isinstance(utt[1], list):
            return utt
        else:
            return default_type, utt
    # no utterance
    if allow_not_an_utterance:
        return None
    raise AssertionError('could not create (typed) utterance from: %s and default_type: %s'
                         % (str(utt), str(default_type)))


def get_data_loaders(args, tokenizer, max_sequence_length=None):
    """ Prepare the dataset(s) for training and evaluation """
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    dataset_paths = args.dataset_path.split(',')
    #background_token_id = tokenizer.special_tokens[TYPE_BACKGROUND]
    # causes strange behaviour (at least in interact.py):
    #bot_token_id = tokenizer.special_tokens.get(TYPE_BOT, tokenizer.special_tokens[TYPE_BOT_DEPRECATED]) if not as_strings else TYPE_BOT
    #user_token_id = tokenizer.special_tokens.get(TYPE_USER, tokenizer.special_tokens[TYPE_USER_DEPRECATED]) if not as_strings else TYPE_USER
    #bot_token_id = tokenizer.special_tokens[TYPE_BOT]
    #user_token_id = tokenizer.special_tokens[TYPE_USER]
    #eos_token_id = tokenizer.special_tokens[TYPE_EOS]
    background_token_id = tokenizer.convert_tokens_to_ids(TYPE_BACKGROUND)
    bot_token_id = tokenizer.convert_tokens_to_ids(TYPE_BOT)
    user_token_id = tokenizer.convert_tokens_to_ids(TYPE_USER)
    eos_token_id = tokenizer.eos_token_id


    for dataset_idx, dataset_path in enumerate(dataset_paths):
        #dataset_id = '<%s>' % os.path.basename(dataset_path)
        #assert dataset_id in tokenizer.special_tokens, \
        #    'dataset_id=%s not found in tokenizer.special_tokens=[%s], but is required as eos token.' \
        #    % (dataset_id, ', '.join(tokenizer.special_tokens.kyes()))
        #dataset_id = tokenizer.special_tokens[dataset_id]

        loaded_dataset = get_dataset(tokenizer, dataset_path, args.dataset_cache)

        logger.info("Build inputs and labels for %s..." % os.path.basename(dataset_path))
        for dataset_name in datasets.keys():
            dataset = loaded_dataset[dataset_name]
            n = 0
            counter_truncated = Counter()
            num_candidates = len(dataset[0]["utterances"][0]["candidates"])
            if args.num_candidates > 0 and dataset_name == 'train':
                num_candidates = min(args.num_candidates, num_candidates)
            for dialog in dataset:
                context = []
                last_context_speaker = None
                if 'background' in dialog:
                    # pass through, if dialog['background'] is a list
                    background = create_typed_utterance(utt=dialog['background'], default_type=background_token_id,
                                                        allow_not_an_utterance=True)
                    if background is not None:
                        context.append(background)
                    else:
                        # assume dialog['background'] is a list of backgrounds
                        for bg in dialog['background']:
                            context.append(create_typed_utterance(utt=bg, default_type=background_token_id))

                if 'personality' in dialog:
                    context.append((bot_token_id, dialog['personality']))
                    if len(context) > 0:
                        last_context_speaker = context[-1][0]

                #for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    # may be None (no personality)
                    current_speaker = last_context_speaker
                    # add speakers to history, if necessary
                    # beginning with user because added personality was from bot
                    # note: iterate alternating over full history to be consistent
                    for i, h in enumerate(utterance["history"]):
                        current_speaker = bot_token_id if current_speaker == user_token_id else user_token_id
                        utterance["history"][i] = create_typed_utterance(utt=utterance["history"][i],
                                                                         default_type=current_speaker)
                    # truncate history
                    history = utterance["history"][-(2*args.max_history+1):]
                    # get previous speaker from history, if available, or take context speaker (may be None)
                    previous_speaker = history[-1][0] if len(history) > 0 else last_context_speaker
                    # switch previous speaker to get current one (default to <uses>, if no context speaker is set)
                    candidate_speaker = bot_token_id if previous_speaker == user_token_id else user_token_id
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        # add speaker, if necessary
                        candidate = create_typed_utterance(utt=candidate, default_type=candidate_speaker)
                        # predict next words only for correct candidate (the last one)
                        lm_labels = bool(j == num_candidates-1)
                        instance, sequence = build_input_from_segments(context, history, candidate,
                                                                       tokenizer, lm_labels,
                                                                       max_sequence_length=max_sequence_length,
                                                                       eos=eos_token_id)
                        l_trunc = len(list(chain(*sequence))) - len(instance['input_ids'])
                        if l_trunc > 0:
                            counter_truncated[l_trunc] += 1
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                        n += 1
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                    if 'adv_labels' in MODEL_INPUTS:
                        datasets[dataset_name]["adv_labels"].append([dataset_idx] * num_candidates)
                    #context = [context[-1]] + context[:-1]  # permuted personalities
            logger.warning('truncated %i of %i instances in %s' % (sum(counter_truncated.values()), n, dataset_name))
            logger.warning('num_trunc_tokens -> frequency: %s' % str(counter_truncated))

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)
        for input_name in MODEL_INPUTS:
            current_data = dataset[input_name]
            tensor = torch.tensor(current_data)
            if input_name not in ["mc_labels", "adv_labels"]:
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="", help="Model type, one of: %s" % ', '.join(MODELS.keys()))
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of a pretrained model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--adv_coef", type=float, default=1.0, help="Adversarial dataset prediction loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    #parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--max_sequence_length", type=int, default=-1, help="If set, use this to manually restrict the sequence length. "
                                                                            "This might be helpful to save resources (memory). "
                                                                            "If not set, this is looked up from the model config (n_ctx value).")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    args.distributed = (args.local_rank != -1)

    logger.info("Prepare tokenizer and data")
    if not args.model:
        logger.warning('"model" parameter is not set! This is deprecated. Please use one of: %s. '
                       'To mimic deprecated behaviour, "model_checkpoint" will be used as "model"' % ', '.join(MODELS.keys()))
        args.model = args.model_checkpoint
    if args.model not in MODELS:
        raise NotImplementedError('model "%s" not implemented. use one of: %s' % (args.model, ', '.join(MODELS.keys())))
    config_class, tokenizer_class, model_class, _ = MODELS[args.model]
    if not args.model_checkpoint:
        args.model_checkpoint = args.model
    # for adversarial training
    #dataset_ids = ['<%s>' % os.path.basename(dataset_path) for dataset_path in args.dataset_path.split(',')]
    model_config = config_class.from_pretrained(args.model_checkpoint)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

    tokenizer.add_special_tokens({'bos_token': TYPE_BOS, 'eos_token': TYPE_EOS, 'pad_token': TYPE_PAD,
                                  'additional_special_tokens': [TYPE_BACKGROUND, TYPE_BOT, TYPE_USER]})

    logger.info("Prepare datasets")
    max_sequence_length = model_config.n_ctx if args.max_sequence_length <= 0 else args.max_sequence_length
    assert max_sequence_length <= model_config.n_ctx, 'max_sequence_length [%i] was set to a value higher than ' \
                                                      'supported by the model (config.n_ctx [%i]). Please use a lower ' \
                                                      'value or do not set it [-1] to use the highest supported one.' \
                                                      % (max_sequence_length, model_config.n_ctx)
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer,
                                                                              max_sequence_length=max_sequence_length)

    logger.info("Prepare pretrained model and optimizer - add special tokens for fine-tuning")

    # Initialize distributed training if needed
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    #model = model_class.from_pretrained(args.model_checkpoint, num_adv_labels=len(dataset_ids))    # for GPT2DoubleHeadsModelwithAdversarial
    model = model_class.from_pretrained(args.model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab


    ####################################################################################################################

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = OpenAIAdam(model.parameters(), lr=args.lr)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    # scheduler is set below (see ignite)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                            num_training_steps=len(train_loader) // args.train_batch_size + 1)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_checkpoint, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_checkpoint, 'scheduler.pt')):
        # Load in optimizer and scheduler states
        # TODO: this needs to be dumped somewhere
        optimizer.load_state_dict(torch.load(os.path.join(args.model_checkpoint, 'optimizer.pt')))
        #scheduler.load_state_dict(torch.load(os.path.join(args.model_checkpoint, 'scheduler.pt')))

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = {MODEL_INPUTS[i]: input_tensor.to(args.device) for i, input_tensor in enumerate(batch)}
        losses = model(**batch)[:2]
        if args.n_gpu > 1: # mean() to average on multi-gpu.
            losses = list(losses)
            for i in range(len(losses)):
                losses[i] = losses[i].mean()
        lm_loss, mc_loss = losses[0], losses[1]
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps

        if 'adv_labels' in MODEL_INPUTS:
            raise NotImplemented
            #loss_wo_adv = loss.clone()
            #if len(losses) > 2:
            #    adv_loss = losses[2]
            #    loss += (adv_loss * args.adv_coef) / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            #scheduler.step()  # Update learning rate schedule # already DONE below!
            optimizer.zero_grad()
        return loss.item(),  #loss_wo_adv.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            #input_ids, mc_token_ids, lm_labels, mc_labels, adv_labels, token_type_ids = batch
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch

            # ATTENTION: requires branch "gpt-2-special-tokens" of "https://github.com/ArneBinder/pytorch-pretrained-BERT"!
            logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()).replace('<pad>', ''))
            model_outputs = model(input_ids=input_ids, mc_token_ids=mc_token_ids, token_type_ids=token_type_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero (scheduler)
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    if 'adv_labels' in MODEL_INPUTS:
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "loss_w/o_adv")
        RunningAverage(output_transform=lambda x: x[0]-x[1]).attach(trainer, "loss_only_adv")
        # TODO: also adapt metrics below for adv_loss?
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        if 'adv_labels' in MODEL_INPUTS:
            tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss_w/o_adv"]), event_name=Events.ITERATION_COMPLETED)
            tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss_only_adv"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        logger.info('save checkpoints to: %s' % tb_logger.writer.log_dir)
        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(tb_logger.writer.log_dir)

        #logger.debug("Saving optimizer and scheduler states to %s", tb_logger.writer.log_dir)
        #torch.save(optimizer.state_dict(), os.path.join(tb_logger.writer.log_dir, 'optimizer.pt'))
        #torch.save(scheduler.state_dict(), os.path.join(tb_logger.writer.log_dir, 'scheduler.pt'))

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    main()
