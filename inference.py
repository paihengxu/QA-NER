import sys
from transformers import pipeline
import torch
from seqeval.metrics import classification_report, f1_score
from collections import defaultdict

from data_utils import parse_and_sample_bio_data_list
from utils.log import logging
from data.label_mapping import conll_mapping
from utils.arguments import params


def get_label_questions(labels, dataset='conll'):
    if dataset == 'conll':
        questions = {'ORG': 'what is the organization?', 'LOC': 'what is the location?',
                     'MISC': 'what is the miscellaneous entity?',
                     'PER': 'who is the person?'}
        logging.info(questions)
        return questions
    questions = {}
    mask_fill = pipeline("fill-mask", model='distilroberta-base', top_k=1,
                         targets=['what', 'who', 'where', 'why', 'when'])

    for label in labels:
        if dataset == 'conll':
            tmp_label = conll_mapping[label]
        else:
            tmp_label = label
        questions[label] = mask_fill(f"<mask> is the {tmp_label.lower().replace('_', ' ').replace('/', ' or ')}?")[0][
            'sequence']
        # questions[label] = f"what is the {tmp_label.lower().replace('_', ' ').replace('/', ' or ')}?"
    logging.info("Label questions:", questions)
    return questions


def convert_prediction_to_bio_labels(qa_pred, words):
    pred_labels = ['O'] * len(words)

    for label, pred_list in qa_pred.items():
        if type(pred_list) != list:
            pred_list = [pred_list]

        # if unanswerable in the highest score, we don't assign any label.
        if pred_list[0]['answer'] == '':
            continue

        start_check = pred_list[0]['start']
        end_check = pred_list[0]['end']
        for ans_idx, pred in enumerate(pred_list):
            if pred['answer'] != '':
                # check for overlaps in repeating examples, keep the answer with highest score
                if ans_idx != 0:
                    if start_check <= pred['start'] <= end_check or start_check <= pred['end'] <= end_check or (
                        pred['start'] < start_check and pred['end'] > end_check):
                        continue

                label_idx_tuple = find_sub_list(pred, words)
                for label_idx in range(label_idx_tuple[0], label_idx_tuple[1]):
                    try:
                        if pred_labels[label_idx] != 'O':
                            print('OVERWRITING')
                            print(qa_pred)
                            print(pred_labels)
                            print("Existing label", pred_labels[label_idx])
                            print("Current label", label)
                            print(words)
                            print("=" * 40)
                    except IndexError:
                        # QA model predicts a subpart of some token
                        # print('TOKENIZATION ERROR (LIKELY)')
                        # print(words)
                        # print(pred)
                        # print(label_idx_tuple)
                        # print(qa_pred)
                        # print("=" * 40)
                        continue
                        # assert 1 == 0

                    if label_idx == label_idx_tuple[0]:
                        pred_labels[label_idx] = f'B-{label}'
                    else:
                        pred_labels[label_idx] = f'I-{label}'
            else:
                if ans_idx != 0:
                    break
    return pred_labels


def find_sub_list(pred, l):
    sent = ' '.join(l)
    if pred['start'] == 0:
        start_idx = 0
    else:
        start_idx = len(sent[:pred['start'] - 1].split(' '))
    end_idx = start_idx + len(pred['answer'].split())

    return start_idx, end_idx


def check_overlap(result):
    """
    filter out the overlap predictions based on confidence score for flat NER
    """
    res = {}
    for label, pred_list in result.items():
        if type(pred_list) != list:
            pred_list = [pred_list]

        # if unanswerable in the highest score, we don't assign any label.
        if pred_list[0]['answer'] == '':
            continue

        start_check = pred_list[0]['start']
        end_check = pred_list[0]['end']
        for ans_idx, pred_dict in enumerate(pred_list):
            if pred_dict['answer'] != '':
                # check for overlaps in repeating examples, keep the answer with highest score
                if ans_idx != 0:
                    if start_check <= pred_dict['start'] <= end_check or \
                        start_check <= pred_dict['end'] <= end_check or \
                        (pred_dict['start'] < start_check and pred_dict['end'] > end_check):
                        continue
            start_check = pred_dict['start']
            end_check = pred_dict['end']
            remove_keys = []
            add_label = True
            for k, v in res.items():
                # if k == 'Rating':
                #     print(k)
                if start_check <= v['start'] <= end_check or start_check <= v['end'] <= end_check or (
                    v['start'] < start_check and v['end'] > end_check):
                    # if previous one has smaller confidence score, then replace
                    if v['score'] < pred_dict['score']:
                        # if previous one has smaller confidence score, then replace
                        remove_keys.append(k)
                    else:
                        # if previous one has larger confidence score, then ignore
                        add_label = False
                        continue
                else:
                    # if no overlap
                    continue
            if add_label:
                res[f"{label}_index_{start_check}_{end_check}"] = pred_dict

            for ele in remove_keys:
                try:
                    del res[ele]
                except KeyError:
                    print('DEBUG')
                    print(res)
                    print(result)
                    print(remove_keys)
                    assert 1 == 0

    final_res = defaultdict(list)
    for k, v in res.items():
        label, start_end = k.split('_index_', 1)
        final_res[label].append(v)
    return final_res


def inference_from_model(model, tokenizer, dataset='conll', debug=None,
                         examples=None, label_set=None):
    if not examples or not label_set:
        if dataset == 'mit':
            examples, label_set = parse_and_sample_bio_data_list('data/restauranttest.bio')
        elif dataset == 'conll':
            examples, label_set = parse_and_sample_bio_data_list('data/conll2003/test.bio')
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    label_questions = get_label_questions(label_set, dataset=dataset)

    device = 0 if torch.cuda.is_available() else -1
    qa = pipeline('question-answering', model=model, max_answer_len=30,
                  tokenizer=tokenizer, device=device, handle_impossible_answer=True)

    true_list = []
    pred_list = []

    for idx, example in enumerate(examples):
        sentence = ' '.join(example.words)
        example_results = {}

        if len(sentence) == 0:
            continue
        for k, v in label_questions.items():
            qa_input = {
                'question': v,
                'context': sentence
            }
            res = qa(qa_input, top_k=20)
            example_results[k] = res

        example_results = check_overlap(example_results)
        label_list = convert_prediction_to_bio_labels(example_results, example.words)

        true_list.append(example.labels)
        pred_list.append(label_list)

        if idx % 500 == 0:
            logging.info(f"Processed {idx} examples")

        if debug and idx > debug:
            break

    print(classification_report(true_list, pred_list))
    return f1_score(true_list, pred_list, average='micro')


def inference(dataset='conll', flat=True,
              model_name="deepset/bert-large-uncased-whole-word-masking-squad2",
              verbose=False):
    logging.info(f"Using Model {model_name}")

    if dataset == 'mit':
        examples, label_set = parse_and_sample_bio_data_list('data/restauranttest.bio')
    elif dataset == 'conll':
        examples, label_set = parse_and_sample_bio_data_list('data/conll2003/test.bio', dataset=dataset)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    label_questions = get_label_questions(label_set, dataset=dataset)

    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")
    qa = pipeline('question-answering', model=model_name, max_answer_len=30,
                  tokenizer=model_name, device=device, handle_impossible_answer=True)

    true_list = []
    pred_list = []

    for idx, example in enumerate(examples):
        sentence = ' '.join(example.words)
        example_results = {}

        # TODO: convert to batch
        if len(sentence) == 0:
            continue
        for k, v in label_questions.items():
            qa_input = {
                'question': v,
                'context': sentence
            }
            res = qa(qa_input, top_k=20)
            example_results[k] = res

        if verbose:
            for k, v in example_results.items():
                if type(v) is not list:
                    v = [v]
                if v[0]['answer'] == '':
                    continue
                else:
                    print(k)
                    print(v)
        if flat:
            example_results = check_overlap(example_results)
        label_list = convert_prediction_to_bio_labels(example_results, example.words)

        if verbose:
            print("PREDICTION", label_list)
            print("LABEL", example.labels)
            print(example_results)
            print('--------------------------------------')

        true_list.append(example.labels)
        pred_list.append(label_list)

        if idx % 500 == 0:
            logging.info(f"Processed {idx} examples")


    print(classification_report(true_list, pred_list))


if __name__ == '__main__':
    dataset = params.dataset
    model_name = params.model_name
    inference(dataset=dataset,
              model_name=model_name,
              verbose=False)
