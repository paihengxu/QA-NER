import json
import random
from collections import defaultdict

from utils.log import logging


class InputExample:
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels


def parse_and_sample_bio_data_list(fn, n=None, test=None, dataset='mit'):
    """
    {label:['O', 'Dish', ......]} and do few-shot sampling
    """
    examples = []
    all_labels = set()  # what are the labels for this dataset
    text2label = {}
    text2words = {}
    with open(fn, 'r') as inf:
        words, labels = [], []
        for idx, line in enumerate(inf):
            if line.strip('\n').startswith('-DOCSTART-'):
                words, labels = [], []
                continue
            if len(line.strip('\n')) == 0:
                if len(words):
                    examples.append(InputExample(words=words, labels=labels))
                    text2label[' '.join(words)] = labels
                    text2words[' '.join(words)] = words
                words, labels = [], []
                if test and idx > test:
                    break
                continue

            if dataset in ['mit', 'conll']:
                label, token = line.strip().split('\t')
                if '-' in label:
                    all_labels.add(label.split('-')[1])
            else:
                raise ValueError(f'Unknown dataset: {dataset}.')
            words.append(token)
            labels.append(label)

        # last example
        if len(words):
            examples.append(InputExample(words=words, labels=labels))
            text2label[' '.join(words)] = labels
            text2words[' '.join(words)] = words

    if n:
        examples = sample_n_per_entity(text2label, text2words, all_labels, n, bio=True)
    logging.info(f'{len(examples)} examples and {len(all_labels)} labels in {fn} are')
    logging.info(all_labels)
    return examples, all_labels


def sample_n_per_entity(text2label, text2word, type_set, n, bio=False):
    # check support examples
    type_count = {_type: 0 for _type in type_set}

    def check_valid():
        valid = [False] * len(type_count)
        for idx, (_type, _curr_count) in enumerate(type_count.items()):
            if _curr_count >= n:
                valid[idx] = True

        return sum(valid) == len(type_count)

    examples = []
    pool = list(text2label.keys())
    logging.info(f'Sampling {n} per entity type for {len(type_count)} entities from {len(pool)} unique text examples')
    while True and len(pool):
        tmp = random.choice(pool)
        tmp_label_covered = set(text2label[tmp])
        pool.remove(tmp)
        if tmp_label_covered == {'O'} or len(tmp_label_covered) == 0:
            continue
        append = [False] * len(tmp_label_covered)
        for idx, _label in enumerate(tmp_label_covered):
            if _label == 'O':
                append[idx] = True
                continue
            else:
                if bio:
                    _label = _label.split('-')[1]
                if type_count[_label] < n:
                    # accept this example
                    append[idx] = True

        if sum(append) == len(tmp_label_covered):
            for _label in tmp_label_covered:
                if _label == 'O':
                    continue
                else:
                    if bio:
                        _label = _label.split('-')[1]
                    type_count[_label] += 1

            examples.append(InputExample(words=text2word[tmp], labels=text2label[tmp]))
        # print(tmp)
        if check_valid():
            break
    logging.info(type_count)
    return examples


if __name__ == '__main__':
    parse_and_sample_bio_data_list('data/conll2003/valid.bio', dataset='conll')
