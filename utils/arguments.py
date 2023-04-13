import argparse


parser = argparse.ArgumentParser(description='Question Answering Based Named Entity Recognition')
parser.add_argument('--dataset', choices=['conll', 'mit'])
# -------------------------------------------------------------------------
# Model params
parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
parser.add_argument('--model-name', type=str, default="deepset/bert-large-uncased-whole-word-masking-squad2")
parser.add_argument('--save-model-name', type=str, default="finetuned")
# -------------------------------------------------------------------------
# Setting params (Hyperparameter)
parser.add_argument('--n', default=5, type=int,
                    help='If 0, use the full dataset. Otherwise, use sampled n examples per entity type.')
parser.add_argument('--num-epochs', default=3, type=int, help='Number of epochs for finetuning the QA model')
parser.add_argument('--num-negatives', default=0, type=int,
                    help='How many negative samples when building the training set')
# -------------------------------------------------------------------------
# Optimization / training params
parser.add_argument('--batch-size', default=4, type=int, help='Batch size (Adjust base on GPU memory)')

params, unparsed = parser.parse_known_args()
