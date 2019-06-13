import json

import spacy


def coqa_split_to_personachat(coqa_data, sentencizer):
    instances = []
    for record in coqa_data:
        instance = {}
        instance['personality'] = sentencizer(record['story'])

        assert len(record['questions']) == len(record['answers']), 'number of questions / answers mismatch'
        instance['utterances'] = []
        history = []
        for i in range(len(record['questions'])):
            utterance = {}
            assert record['questions'][i]['turn_id'] == record['answers'][i]['turn_id'] == i + 1, 'turn_id mismatch'
            question_text = record['questions'][i]['input_text']
            answer_text = record['answers'][i]['input_text']
            history.append(question_text)
            utterance['history'] = history.copy()
            utterance['candidates'] = [answer_text]
            instance['utterances'].append(utterance)
            history.append(answer_text)

        instances.append(instance)

    return instances


def coqa_to_personachat(coqa_dev='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-dev-v1.0.json',
                        coqa_train='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-train-v1.0.json',
                        out='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa_converted_persona.json'):
    sentencizer = create_sentencizer()
    print('convert dev...')
    coqa_dev = json.load(open(coqa_dev))['data']
    coqa_converted_dev = coqa_split_to_personachat(coqa_data=coqa_dev, sentencizer=sentencizer)
    print('convert train...')
    coqa_train = json.load(open(coqa_train))['data']
    coqa_converted_train = coqa_split_to_personachat(coqa_data=coqa_train, sentencizer=sentencizer)
    json.dump({'train': coqa_converted_train, 'valid': coqa_converted_dev},
              open(out, 'w'), indent=2)


def gen_personachat_extract(fn, extract_size=10):
    data = json.load(open(fn))

    fn_out = fn.replace('.json', '_extract%i.json' % extract_size)

    # print dataset size
    for k in data:
        print('%s: %i' % (k, len(data[k])))
    print('write to: %s' % fn_out)
    json.dump({k: data[k][:extract_size] for k in data}, open(fn_out, 'w'), indent=2)


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
    #gen_personachat_extract(fn='/mnt/DATA/ML/data/corpora/dialog/personachat_self_original.json')

    # convert CoQA to personachat
    coqa_to_personachat()
    # stats: train: 7199; valid: 500
    gen_personachat_extract(fn='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa_converted_persona.json')

    #x = dummy_tokenize()

    print('done')
