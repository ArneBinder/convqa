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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, GPT2DoubleHeadsModel,
                                     GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, GPT2LMHeadModel, OpenAIGPTLMHeadModel)

from utils import get_dataset

# NOTE: the padding token has to be at SPECIAL_TOKENS[-1]!
SPECIAL_TOKENS = ["<bos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

# implemented models:
#   <model name> -> (<Tokenizer class>, <DoubleHeadsModel class>, <LMHeadModel>)
# see huggingface/pytorch-pretrained-bert for available model names
MODELS = {
    'openai-gpt': (OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel),
    'gpt2': (GPT2Tokenizer, GPT2DoubleHeadsModel, GPT2LMHeadModel),
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
    #if 0 < max_sequence_length < max_l:
    #    bigger_l = {k: l_counter[k] for k in l_counter.keys() if k > max_sequence_length}
    #    logger.warning('%i of %i entries exceed max_sequence_length=%i (these inputs will be truncated): \n%s'
    #                   % (sum(bigger_l.values()), len(dataset["input_ids"]), max_sequence_length, bigger_l))

    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(context, history, reply, tokenizer, lm_labels=False, eos=None, return_strings=False, max_sequence_length=None):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    if return_strings:
        bos, speaker1, speaker2 = SPECIAL_TOKENS[:-1]
    else:
        bos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    instance = {}
    # create sequence list: [bos+persona, history0, ..., historyN, reply+(eos)]
    sequence = [[bos] + list(chain(*context))] + history + [reply + ([eos] if eos is not None else [])]
    # prepend speaker1/2 to history entries and current reply:
    #   [bos+persona, speaker1+history0, ..., speaker2+historyN, speaker1+reply+(eos)]
    sequence = [sequence[0]] + [[speaker2 if (i + 1) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    #sequence_deprecated = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    #assert all([sequence_deprecated[i] == sequence[i] for i in range(len(sequence))]), 'mismatch'

    instance["input_ids"] = list(chain(*sequence))
    # set persona and speaker1 utterances to speaker1-type and set speaker2 utterances to speaker2-type
    #instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["token_type_ids"] = [s[0] for i, s in enumerate(sequence) for _ in s]
    if max_sequence_length:
        instance["input_ids"] = instance["input_ids"][:max_sequence_length]
        instance["token_type_ids"] = instance["token_type_ids"][:max_sequence_length]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        if max_sequence_length:
            instance["lm_labels"] = instance["lm_labels"][:max_sequence_length]
    return instance, sequence


def get_data_loaders(args, tokenizer, as_strings=False, max_sequence_length=None):
    """ Prepare the dataset(s) for training and evaluation """
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    dataset_paths = args.dataset_path.split(',')
    for dataset_path in dataset_paths:
        dataset_id = '<%s>' % os.path.basename(dataset_path)
        assert dataset_id in tokenizer.special_tokens, \
            'dataset_id=%s not found in tokenizer.special_tokens=[%s], but is required as eos token.' \
            % (dataset_id, ', '.join(tokenizer.special_tokens.kyes()))
        dataset_id_converted = tokenizer.special_tokens[dataset_id] if not as_strings else dataset_id
        loaded_dataset = get_dataset(tokenizer, dataset_path, args.dataset_cache, as_strings=as_strings)

        logger.info("Build inputs and labels for %s..." % os.path.basename(args.dataset_path))
        for dataset_name in datasets.keys():
            dataset = loaded_dataset[dataset_name]
            n = 0
            counter_truncated = Counter()
            num_candidates = len(dataset[0]["utterances"][0]["candidates"])
            if args.num_candidates > 0 and dataset_name == 'train':
                num_candidates = min(args.num_candidates, num_candidates)
            for dialog in dataset:
                context = dialog["personality"] #.copy()
                #for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance, sequence = build_input_from_segments(context, history, candidate, tokenizer, lm_labels,
                                                                       return_strings=as_strings,
                                                                       max_sequence_length=max_sequence_length,
                                                                       eos=dataset_id_converted)
                        l_trunc = len(list(chain(*sequence))) - len(instance['input_ids'])
                        if l_trunc > 0:
                            counter_truncated[l_trunc] += 1
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                        n += 1
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                    #context = [context[-1]] + context[:-1]  # permuted personalities
            logger.warning('truncated %i of %i instances in %s' % (sum(counter_truncated.values()), n, dataset_name))
            logger.warning('num_trunc_tokens -> frequency: %s' % str(counter_truncated))

    assert not as_strings, 'return_strings has to be False'

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            x = dataset[input_name]
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
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
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
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

    n_gpu = 0

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        n_gpu = 1
    elif args.device == 'cuda':
        n_gpu = torch.cuda.device_count()

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    if not args.model:
        logger.warning('"model" parameter is not set! This is deprecated. Please use one of: %s. '
                       'To mimic deprecated behaviour, "model_checkpoint" will be used as "model"' % ', '.join(MODELS.keys()))
        args.model = args.model_checkpoint
    if args.model not in MODELS:
        raise NotImplementedError('model "%s" not implemented. use one of: %s' % (args.model, ', '.join(MODELS.keys())))
    tokenizer_class, model_class, _ = MODELS[args.model]
    if not args.model_checkpoint:
        args.model_checkpoint = args.model
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    dataset_ids = ['<%s>' % os.path.basename(dataset_path) for dataset_path in args.dataset_path.split(',')]
    tokenizer.set_special_tokens(SPECIAL_TOKENS + dataset_ids)
    model.set_num_special_tokens(len(tokenizer.special_tokens))
    model.to(args.device)
    optimizer = OpenAIAdam(model.parameters(), lr=args.lr)
    model_config = model.config

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)  # device_ids will include all GPU devices by default
        logger.info('train on %i GPUs' % n_gpu)

    logger.info("Prepare datasets")
    max_sequence_length = model_config.n_ctx if args.max_sequence_length <= 0 else args.max_sequence_length
    assert max_sequence_length <= model_config.n_ctx, 'max_sequence_length [%i] was set to a value higher than ' \
                                                      'supported by the model (config.n_ctx [%i]). Please use a lower ' \
                                                      'value or do not set it [-1] to use the highest supported one.' \
                                                      % (max_sequence_length, model_config.n_ctx)
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer, #as_strings=True,
                                                                              max_sequence_length=max_sequence_length)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        lm_loss, mc_loss = model(*batch)
        if n_gpu > 1:
            lm_loss = lm_loss.mean()  # mean() to average on multi-gpu.
            mc_loss = mc_loss.mean()
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            # ATTENTION: requires branch "gpt-2-special-tokens" of "https://github.com/ArneBinder/pytorch-pretrained-BERT"!
            logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()).replace('<pad>', ''))
            model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
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

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
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
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        logger.info('save checkpoints to: %s' % tb_logger.writer.log_dir)
        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
