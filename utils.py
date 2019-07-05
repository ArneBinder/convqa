# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import pickle
import re
import tarfile
import tempfile

import requests
import torch

from pytorch_pretrained_bert import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

logger = logging.getLogger(__file__)

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()

    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def get_dataset(tokenizer, dataset_path, dataset_cache=None, as_strings=False):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + dataset_path.split('/')[-1].replace('.json', '') + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if as_strings:
        dataset_cache += '_STRINGS'

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                # remove space before sentence marker and tokenize
                tokens = tokenizer.tokenize(obj.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','))
                if as_strings:
                    return tokens
                return tokenizer.convert_tokens_to_ids(tokens)
            if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
                speaker, obj = obj
                if as_strings:
                    return speaker, tokenize(obj)
                return tokenizer.special_tokens[speaker], tokenize(obj)
            if isinstance(obj, dict):
                return dict((n, tokenize(' '.join(o))) if n == 'personality' and isinstance(o, list) else (n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + dataset_path.split('/')[-1].replace('.json', '') + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)

    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


    def context_fetcher(s, previous_context=None):
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
                            res[wikipedia_entity_uri] = [s.strip() for s in wikipedia_entry['text']]
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
    return context_fetcher
