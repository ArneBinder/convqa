import bz2
import json
import logging
import os
import pickle
import re
import tarfile
from collections import Counter

import numpy as np
import requests
import spacy
from tqdm import tqdm

logger = logging.getLogger(__file__)

# too large for memory
def sample_neg_indices_old(n_instances, n_candidates):
    # create index array [[0, 1, .., n_instances-1], .., [0, 1, .., n_instances-1]]
    a = np.tile(np.arange(n_instances), n_instances).reshape((n_instances, n_instances))
    # for each row, replace current idx with last
    np.fill_diagonal(a, n_instances-1)
    # truncate replaced index (last one)
    a = a[:, :-1]
    # shuffle each row
    #np.random.shuffle(a.T)
    np.apply_along_axis(np.random.shuffle, axis=1, arr=a)
    # return first n_candidates of each row
    return a[:, :n_candidates]


def sample_neg_candidates(instances, n_candidates, n_resample=3):
    if not isinstance(instances, np.ndarray):
        instances = np.array(instances)

    n_collisions = 0
    nn_collisions = 0

    a = np.empty(shape=(len(instances), n_candidates - 1), dtype=instances.dtype)
    for i, inst in enumerate(instances):
        i_sample = 0
        a[i] = np.random.choice(instances,  n_candidates - 1)

        # check for collisions with correct instance
        # NOTE: we do not normalize the case (e.g. of the first character)!
        collision_indices = np.nonzero(a[i] == inst)[0]
        while len(collision_indices) > 0 and i_sample < n_resample:
            new_samples = np.random.choice(instances,  len(collision_indices))
            a[i][collision_indices] = new_samples
            collision_indices = np.nonzero(a[i] == inst)[0]
            i_sample += 1
        if len(collision_indices) > 0:
            nn_collisions += 1
            n_collisions += len(collision_indices)

    logger.info('collisions: %i (in %i instances; total: %i)' % (n_collisions, nn_collisions, len(instances)))
    return a

def count_sentences(s, sentencizer, counter=None):
    s = s.strip()
    try:
        sents = sentencizer(s)
    except Exception as e:
        if ' ' in s:
            logger.warning('could not sentencize "%s", return as ONE sentence (%s)' % (s.strip(), e))
        sents = [s]
    if counter is not None:
        counter[len(sents)] +=1
    return len(sents)

def create_instance(record, sentencizer, max_sentences_qa, max_sentences_persona, stats):
    all_questions = []
    all_answers = []

    was_truncated = False
    instance = {}
    instance['context'] = record['story']
    if max_sentences_persona is not None:
        stats['persona']['n_sents'][len(instance['context'])] += 1
        if len(instance['context']) > max_sentences_persona:
            was_truncated = True
        instance['context'] = instance['context'][:max_sentences_persona]

    assert len(record['questions']) == len(record['answers']), 'number of questions / answers mismatch'
    #instance['utterances'] = []
    instance['n_utterances'] = 0
    #history = []
    for i in range(len(record['questions'])):
        #utterance = {}
        assert record['questions'][i]['turn_id'] == record['answers'][i]['turn_id'] == i + 1, 'turn_id mismatch'
        question_text = record['questions'][i]['input_text']
        answer_text = record['answers'][i]['input_text']
        # skip answer-question pairs if number of sentences in one of them > max_sentences
        continue_this = False
        if max_sentences_qa and count_sentences(s=question_text, sentencizer=sentencizer,
                                                counter=stats['question']['n_sents']) > max_sentences_qa:
            continue_this = True
        if max_sentences_qa and count_sentences(s=answer_text, sentencizer=sentencizer,
                                                counter=stats['answer']['n_sents']) > max_sentences_qa:
            continue_this = True
        if continue_this:
            was_truncated = True
            continue

        #history.append(question_text)
        #utterance['history'] = history.copy()
        all_answers.append(answer_text)
        all_questions.append(question_text)
        #instance['utterances'].append(utterance)
        #history.append(answer_text)
        instance['n_utterances'] += 1

    return instance, all_questions, all_answers, was_truncated

def coqa_split_to_personachat(coqa_data, sentencizer, n_candidates=20, max_sentences_qa=1, max_sentences_context=None,
                              create_question_utterances=False):
    instances = []
    all_answers = []
    all_questions = []
    stats = {'context': {'n_sents': Counter()}, 'answer': {'n_sents': Counter()}, 'question': {'n_sents': Counter()}}
    n_skipped = 0
    for record in coqa_data:
        instance, current_questions, current_answers, was_truncated = create_instance(
            record=record, sentencizer=sentencizer, max_sentences_qa=max_sentences_qa,
            max_sentences_persona=max_sentences_context, stats=stats)
        if was_truncated:
            n_skipped += 1
            continue
        instances.append(instance)
        all_questions.extend(current_questions)
        all_answers.extend(current_answers)
    logger.info('data created (skipped %i out of %i)' % (n_skipped, len(coqa_data)))
    logger.info('max_sentences_persona: %s' % str(max_sentences_context))
    logger.info('max_sentences_qa: %s' % str(max_sentences_qa))
    logger.info(stats)

    sampled_neg_answers = sample_neg_candidates(instances=all_answers, n_candidates=n_candidates)
    sampled_neg_questions = None
    if create_question_utterances:
        sampled_neg_questions = sample_neg_candidates(instances=all_questions, n_candidates=n_candidates)

    logger.info('neg samples created')
    #all_candidates = np.concatenate([sampled_neg_answers.T, [all_answers]]).T

    i = 0
    for instance in instances:
        instance['utterances'] = []
        history = []
        #for j, utterance in enumerate(instance['utterances']):
        for _ in range(instance['n_utterances']):
            if sampled_neg_questions is not None:
                new_utterance = {'history': history.copy(),
                                 'candidates': sampled_neg_questions[i].tolist() + [all_questions[i]]}
                instance['utterances'].append(new_utterance)
            history.append(all_questions[i])

            new_utterance = {'history': history.copy(),
                             'candidates': sampled_neg_answers[i].tolist() + [all_answers[i]]}
            instance['utterances'].append(new_utterance)
            history.append(all_answers[i])
            i += 1
        del instance['n_utterances']
    logger.info('candidates created')

    return instances


def coqa_to_personachat(coqa_dev='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-dev-v1.0.json',
                        coqa_train='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-train-v1.0.json',
                        out=None,
                        n_candidates=20,
                        max_sents_qa=1,
                        max_sents_context=None,
                        create_question_utterances=False):
    if out is None:
        fn = 'coqa_converted_persona'
        if max_sents_qa and max_sents_qa >= 0:
            fn += '_sentsqa%i' % max_sents_qa
        if max_sents_context and max_sents_context >= 0:
            fn += '_sentsp%i' % max_sents_context
        if create_question_utterances:
            fn += '_questionutterances'

        out = os.path.join(os.path.dirname(coqa_train), '%s.json' % fn)

    sentencizer = create_sentencizer()
    coqa_converted = {}
    #print('convert dev...')
    coqa_dev = json.load(open(coqa_dev))['data']
    coqa_converted['valid'] = coqa_split_to_personachat(coqa_data=coqa_dev, sentencizer=sentencizer,
                                                        n_candidates=n_candidates, max_sentences_qa=max_sents_qa,
                                                        max_sentences_context=max_sents_context,
                                                        create_question_utterances=False)
    if create_question_utterances:
        coqa_converted['valid_questionutterances'] = coqa_split_to_personachat(coqa_data=coqa_dev, sentencizer=sentencizer,
                                                                               n_candidates=n_candidates, max_sentences_qa=max_sents_qa,
                                                                               max_sentences_context=max_sents_context,
                                                                               create_question_utterances=True)
    logger.info('convert train...')
    coqa_train = json.load(open(coqa_train))['data']
    coqa_converted['train'] = coqa_split_to_personachat(coqa_data=coqa_train, sentencizer=sentencizer,
                                                        n_candidates=n_candidates, max_sentences_qa=max_sents_qa,
                                                        max_sentences_context=max_sents_context,
                                                        create_question_utterances=create_question_utterances)

    logger.info('dump to json: %s ...' % out)
    json.dump(coqa_converted, open(out, 'w'), indent=2)
    return out


def gen_personachat_extract(fn, extract_size=10, start_idx=0):
    data = json.load(open(fn))
    if start_idx > 0:
        fn_out = fn.replace('.json', '_extract%s_start%s.json' % (str(extract_size), str(start_idx)))
    else:
        fn_out = fn.replace('.json', '_extract%s.json' % str(extract_size))

    # print dataset size
    for k in data:
        logger.info('%s: %i' % (k, len(data[k])))
    logger.info('write to: %s' % fn_out)
    if extract_size is not None:
        data = {k: data[k][start_idx:extract_size+start_idx] for k in data}
    json.dump(data, open(fn_out, 'w'), indent=2)


def create_sentencizer(spacy_model='en_core_web_sm'):
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


def convert_hotpotqa_wikidump_to_dict(fn, fields=('text', 'title')):
    entries = {}
    with tarfile.open(fn, "r:bz2") as tar:
        for tarinfo in tqdm(tar):
            f = tar.extractfile(tarinfo)
            if f is not None:
                uncomp = bz2.decompress(f.read())
                for l in uncomp.split(b'\n'):
                    if l.strip() != b'':
                        entry = json.loads(l)
                        entries[int(entry['id'])] =  {f: entry[f] for f in fields}
    return entries


def dummy_tokenize():
    from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    # Tokenized input
    text = "Who was Jim Henson ? Jim Henson was a puppeteer"
    tokenized_text = tokenizer.tokenize(text)
    return tokenized_text


if __name__ == '__main__':
    #stats: train: 17878; valid: 1000
    #gen_personachat_extract(fn='/mnt/DATA/ML/data/corpora/dialog/personachat_self_original.json', extract_size=10)

    # convert CoQA to personachat
    out_fn = coqa_to_personachat(max_sents_context=None, create_question_utterances=False)
    # stats: train: 7199; valid: 500
    gen_personachat_extract(fn=out_fn, extract_size=10, start_idx=0)

    #x = dummy_tokenize()

    logger.info('done')
