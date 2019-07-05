import bz2
import json
import logging
import os
import tarfile
from collections import Counter

import numpy as np
from tqdm import tqdm

from interact import create_sentencizer

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

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

def create_instance_from_coqa(record, stats, sentencizer=None, max_sentences_qa=1, max_sentences_background=None):
    all_questions = []
    all_answers = []

    was_truncated = False
    instance = {}
    instance['background'] = record['story']
    if max_sentences_background is not None:
        if 'background' not in stats:
            stats['background'] = {'n_sents': Counter()}
        stats['background']['n_sents'][len(instance['background'])] += 1
        if len(instance['background']) > max_sentences_background:
            was_truncated = True
        instance['background'] = instance['background'][:max_sentences_background]

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
        if sentencizer is not None:
            if 'question' not in stats:
                stats['question'] = {'n_sents': Counter()}
            if max_sentences_qa and count_sentences(s=question_text, sentencizer=sentencizer,
                                                    counter=stats['question']['n_sents']) > max_sentences_qa:
                continue_this = True
            if 'answer' not in stats:
                stats['answer'] = {'n_sents': Counter()}
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

def dataset_split_to_dialog(data, instance_builder=create_instance_from_coqa, n_candidates=20,
                            create_question_utterances=False, **instance_builder_kargs
                            ):
    instances = []
    all_answers = []
    all_questions = []
    stats = {}
    n_skipped = 0
    for record in data:
        instance, current_questions, current_answers, was_truncated = instance_builder(
            record=record, stats=stats, **instance_builder_kargs)
        if was_truncated:
            n_skipped += 1
            continue
        instances.append(instance)
        all_questions.extend(current_questions)
        all_answers.extend(current_answers)
    logger.info('data created (skipped %i out of %i)' % (n_skipped, len(data)))
    #logger.info('max_sentences_background: %s' % str(max_sentences_background))
    #logger.info('max_sentences_qa: %s' % str(max_sentences_qa))
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


def convert_to_dialog(dir='/mnt/DATA/ML/data/corpora/QA/CoQA',
                      dev='coqa-dev-v1.0.json',
                      train='coqa-train-v1.0.json',
                      out=None,
                      n_candidates=20,
                      create_question_utterances=False,
                      instance_builder=create_instance_from_coqa,
                      **instance_builder_kwargs
                      ):
    dev = os.path.join(dir, dev)
    train = os.path.join(dir, train)
    if out is None:
        dataset_name = os.path.basename(dir) or os.path.basename(os.path.dirname(dir))
        fn = '%s_converted_dialog' % dataset_name.lower()
        if instance_builder_kwargs.get('max_sentences_qa', -1) >= 0:
            fn += '_sentsqa%i' % instance_builder_kwargs['max_sentences_qa']
        if instance_builder_kwargs.get('max_sentences_background', -1) >= 0:
            fn += '_sentsb%i' % instance_builder_kwargs['max_sentences_background']
        if create_question_utterances:
            fn += '_questionutterances'

        out = os.path.join(dir, '%s.json' % fn)


    converted = {}
    #print('convert dev...')
    dev = json.load(open(dev))['data']
    converted['valid'] = dataset_split_to_dialog(data=dev, n_candidates=n_candidates, instance_builder=instance_builder,
                                                 create_question_utterances=False, **instance_builder_kwargs)
    if create_question_utterances:
        converted['valid_questionutterances'] = dataset_split_to_dialog(data=dev, n_candidates=n_candidates,
                                                                        instance_builder=instance_builder,
                                                                        create_question_utterances=True,
                                                                        **instance_builder_kwargs)
    logger.info('convert train...')
    train = json.load(open(train))['data']
    converted['train'] = dataset_split_to_dialog(data=train, n_candidates=n_candidates, instance_builder=instance_builder,
                                                 create_question_utterances=create_question_utterances,
                                                 **instance_builder_kwargs)

    logger.info('dump to json: %s ...' % out)
    json.dump(converted, open(out, 'w'), indent=2)
    return out


def gen_dataset_extract(fn, extract_size=10, start_idx=0):
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


def convert_coqa(create_question_utterances=True, max_sentences_qa=1, sentencizer=create_sentencizer()):
    # convert CoQA to conversational QA format
    return convert_to_dialog(dir='/mnt/DATA/ML/data/corpora/QA/CoQA',
                             dev='coqa-dev-v1.0.json',
                             train='coqa-train-v1.0.json',
                             out=None,
                             instance_builder=create_instance_from_coqa, max_sentences_qa=max_sentences_qa,
                             create_question_utterances=create_question_utterances, sentencizer=sentencizer)
    # stats: train: 7199; valid: 500



if __name__ == '__main__':
    #stats: train: 17878; valid: 1000
    #gen_personachat_extract(fn='/mnt/DATA/ML/data/corpora/dialog/personachat_self_original.json', extract_size=10)

    out_fn = convert_coqa()


    gen_dataset_extract(fn=out_fn, extract_size=10, start_idx=0)
    #x = dummy_tokenize()

    logger.info('done')