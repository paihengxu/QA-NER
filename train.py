import sys
import transformers
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer, pipeline
from transformers import set_seed
import torch
import random
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from data_utils import parse_and_sample_bio_data_list
from inference import get_label_questions
from utils.log import logging
from utils.arguments import params
from inference import inference_from_model

random.seed(123)
set_seed(42)

# model_checkpoint = "deepset/bert-large-uncased-whole-word-masking-squad2"
model_checkpoint = params.model_name
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
batch_size = params.batch_size
max_length = 384  # The maximuqm length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.

pad_on_right = tokenizer.padding_side == "right"


def load_ner_datasets(dataset, example_per_entity=None, entity_split=False, num_negative=None):

    if dataset == 'mit':
        dev_examples, dev_types = parse_and_sample_bio_data_list('data/restauranttest.bio', n=example_per_entity)
        train_examples, train_types = parse_and_sample_bio_data_list('data/restauranttrain.bio', n=example_per_entity)
        assert len(dev_types) == len(train_types)
        label_set = train_types
    elif dataset == 'conll':
        dev_examples, dev_types = parse_and_sample_bio_data_list('data/conll2003/valid.bio', n=None,
                                                                 dataset='conll')
        train_examples, train_types = parse_and_sample_bio_data_list('data/conll2003/train.bio', n=example_per_entity,
                                                                     dataset='conll')
        # dev_examples, dev_types = train_examples, train_types
        assert len(dev_types) == len(train_types)
        label_set = train_types
    else:
        raise NotImplementedError

    data = {'train': train_examples, 'dev': dev_examples}
    label_types = {'train': train_types, 'dev': dev_types}
    # label_questions = get_label_questions(label_set, dataset=dataset)
    label_questions = get_label_questions(label_set, dataset=dataset)
    result_dict = defaultdict(list)
    result = dict()

    idx = 0
    for fold, v in data.items():
        for example_idx, example in enumerate(v):
            sentence = ' '.join(example.words)

            # get answers from tag list, e.g., ['O', 'O', 'O', 'organization-education', 'O', 'O', 'O', 'O']
            answers = defaultdict(list)
            tmp_ans = []
            prev_tag = 'O'
            prev_words = []
            for word, tag in zip(example.words+[None], example.labels+['O']):
                if tag == 'O' or (tag != prev_tag and prev_tag != 'O'):
                    if len(tmp_ans):
                        _, fine_label = prev_tag.strip().split('-')
                        current_fine_label = tag.strip().split('-')[1] if tag != 'O' else 'O'

                        if fine_label != current_fine_label:
                            # list of (answer, answer_start_idx)
                            start_pos = len(' '.join(prev_words[:-len(tmp_ans)]))
                            start_pos = start_pos + 1 if start_pos != 0 else start_pos
                            answers[fine_label].append((' '.join(tmp_ans), start_pos))
                            tmp_ans = [] if current_fine_label == 'O' else [word]
                        else:
                            tmp_ans.append(word)
                else:
                    tmp_ans.append(word)
                prev_tag = tag
                prev_words.append(word)
            # print(example.words)
            # print(example.labels)
            # print(answers)
            tmp_negatives = []
            for label, question in label_questions.items():
                if entity_split and label.replace(' ', '') not in label_types[fold]:
                    continue
                if label in answers:
                    for ans_tuple in answers[label]:
                        tmp_q = dict()
                        tmp_q['id'] = idx
                        tmp_q['context'] = sentence
                        tmp_q['question'] = question

                        answer_text, answer_span_start = ans_tuple
                        tmp_q['answers'] = {'answer_start': [answer_span_start], 'text': [answer_text]}

                        assert len(tmp_q) == 4
                        result_dict[fold].append(tmp_q)
                        idx += 1
                else:
                    # negative examples
                    tmp_q = dict()
                    tmp_q['id'] = idx
                    tmp_q['context'] = sentence
                    tmp_q['question'] = question
                    tmp_q['answers'] = {'answer_start': [-1], 'text': ['']}
                    tmp_negatives.append(tmp_q)
                    idx += 1
            if num_negative:
                result_dict[fold].extend(random.sample(tmp_negatives, num_negative))
            else:
                result_dict[fold].extend(tmp_negatives)

        df = pd.DataFrame(result_dict[fold])
        df['q_len'] = df['question'].apply(len)
        df['context_len'] = df['context'].apply(len)
        result[fold] = Dataset.from_pandas(df)

        logging.info(f"Number of {fold} examples: {len(df)}")
    logging.info("Finished preparing dataset...")
    return DatasetDict(result)


def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                try:
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                except IndexError:
                    # print(examples)
                    print(token_end_index)
                    sys.exit(1)
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second" if pad_on_right else "only_first",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


if __name__ == '__main__':
    n = params.n
    num_negatives = params.num_negatives

    dataset = params.dataset
    datasets = load_ner_datasets(dataset=dataset, example_per_entity=n, num_negative=num_negatives)

    for i in range(20):
        print(datasets["train"][i])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using {device}")
    tokenized_datasets = datasets.map(prepare_train_features, batched=True,
                                      remove_columns=datasets["train"].column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)

    model_name = model_checkpoint.split("/")[-1]
    save_model_name = params.save_model_name
    # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/trainer#transformers.TrainingArguments
    args = TrainingArguments(
        save_model_name,
        overwrite_output_dir=True,
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=params.num_epochs,
        # weight_decay=0.001,
        push_to_hub=False,
        # save_strategy='epoch',
        save_strategy='no',
    )
    logging.info(f"Trainer using {args.device}")


    from transformers import default_data_collator

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    inference_from_model(model, tokenizer, dataset=dataset)
