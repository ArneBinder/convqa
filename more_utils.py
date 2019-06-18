import json
import os
from collections import Counter

import numpy as np
import spacy

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

    print('collisions: %i (in %i instances; total: %i)' % (n_collisions, nn_collisions, len(instances)))
    return a

def count_sentences(s, sentencizer, counter=None):
    s = s.strip()
    try:
        sents = sentencizer(s)
    except Exception as e:
        if ' ' in s:
            print('WARNING: could not sentencize "%s", return as ONE sentence (%s)' % (s.strip(), e))
        sents = [s]
    if counter is not None:
        counter[len(sents)] +=1
    return len(sents)

def create_instance(record, sentencizer, max_sentences_qa, max_sentences_persona, stats):
    all_answers = []

    was_truncated = False
    instance = {}
    instance['personality'] = sentencizer(record['story'])
    if max_sentences_persona is not None:
        stats['persona']['n_sents'][len(instance['personality'])] += 1
        if len(instance['personality']) > max_sentences_persona:
            was_truncated = True
        instance['personality'] = instance['personality'][:max_sentences_persona]

    assert len(record['questions']) == len(record['answers']), 'number of questions / answers mismatch'
    instance['utterances'] = []
    history = []
    for i in range(len(record['questions'])):
        utterance = {}
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

        history.append(question_text)
        utterance['history'] = history.copy()
        # utterance['candidates'] = [answer_text]
        all_answers.append(answer_text)
        instance['utterances'].append(utterance)
        history.append(answer_text)

    return instance, all_answers, was_truncated

def coqa_split_to_personachat(coqa_data, sentencizer, n_candidates=20, max_sentences_qa=1, max_sentences_persona=None):
    instances = []
    all_answers = []
    stats = {'persona':{'n_sents': Counter()}, 'answer': {'n_sents': Counter()}, 'question': {'n_sents': Counter()}}
    n_skipped = 0
    for record in coqa_data:
        instance, current_answers, was_truncated = create_instance(record=record, sentencizer=sentencizer,
                                                        max_sentences_qa=max_sentences_qa,
                                                        max_sentences_persona=max_sentences_persona, stats=stats)
        if was_truncated:
            n_skipped += 1
            continue
        instances.append(instance)
        all_answers.extend(current_answers)
    print('data created (skipped %i out of %i)' % (n_skipped, len(coqa_data)))
    print('max_sentences_persona: %s' % str(max_sentences_persona))
    print('max_sentences_qa: %s' % str(max_sentences_qa))
    print(stats)

    all_answers = np.array(all_answers)
    sampled_neg_answers = sample_neg_candidates(instances=all_answers, n_candidates=n_candidates)
    #all_sampled_answers = all_answers[sampled_neg_indices]
    print('neg samples created')
    #all_candidates = np.concatenate([sampled_neg_answers.T, [all_answers]]).T

    i = 0
    for instance in instances:
        for j, utterance in enumerate(instance['utterances']):
            # prepend samples
            instance['utterances'][j]['candidates'] = sampled_neg_answers[i].tolist() + [all_answers[i]]
            i += 1
    print('candidates created')

    return instances


def coqa_to_personachat(coqa_dev='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-dev-v1.0.json',
                        coqa_train='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-train-v1.0.json',
                        out=None,
                        n_candidates=20,
                        max_sents_qa=1,
                        max_sents_persona=2):
    if out is None:
        fn = 'coqa_converted_persona'
        if max_sents_qa and max_sents_qa >= 0:
            fn += '_sentsqa%i' % max_sents_qa
        if max_sents_persona and max_sents_persona >= 0:
            fn += '_sentsp%i' % max_sents_persona
        out = os.path.join(os.path.dirname(coqa_train), '%s.json' % fn)

    sentencizer = create_sentencizer()
    #print('convert dev...')
    coqa_dev = json.load(open(coqa_dev))['data']
    coqa_converted_dev = coqa_split_to_personachat(coqa_data=coqa_dev, sentencizer=sentencizer,
                                                   n_candidates=n_candidates, max_sentences_qa=max_sents_qa,
                                                   max_sentences_persona=max_sents_persona)
    print('convert train...')
    coqa_train = json.load(open(coqa_train))['data']
    coqa_converted_train = coqa_split_to_personachat(coqa_data=coqa_train, sentencizer=sentencizer,
                                                     n_candidates=n_candidates, max_sentences_qa=max_sents_qa,
                                                     max_sentences_persona=max_sents_persona)
    print('dump to json: %s ...' % out)
    json.dump({'train': coqa_converted_train,
               'valid': coqa_converted_dev
               },
              open(out, 'w'), indent=2)


def gen_personachat_extract(fn, extract_size=10, start_idx=0):
    data = json.load(open(fn))
    if start_idx > 0:
        fn_out = fn.replace('.json', '_extract%s_start%s.json' % (str(extract_size), str(start_idx)))
    else:
        fn_out = fn.replace('.json', '_extract%s.json' % str(extract_size))

    # print dataset size
    for k in data:
        print('%s: %i' % (k, len(data[k])))
    print('write to: %s' % fn_out)
    if extract_size is not None:
        data = {k: data[k][start_idx:extract_size+start_idx] for k in data}
    json.dump(data, open(fn_out, 'w'), indent=2)


def create_sentencizer():
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    #sentencizer = lambda s: [sent.text for sent in nlp(s.strip(), disable=['parser', 'tagger', 'ner']).sents]
    def sentencizer(s):
        sents = []
        for sent in nlp(s.strip(), disable=['parser', 'tagger', 'ner']).sents:
            sents.extend([_sent.strip() for _sent in sent.text.split('\n\n') if _sent.strip() != ''])
        return sents
    return sentencizer


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
    #coqa_to_personachat(max_sents_persona=20)
    # stats: train: 7199; valid: 500
    #gen_personachat_extract(fn='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa_converted_persona_sentsqa1_sentsp20.json', extract_size=100, start_idx=0)

    #x = dummy_tokenize()

    print('done')
