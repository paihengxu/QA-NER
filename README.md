# QA-NER

An unofficial implementation of QANER

## Install
The repo was tested with Python 3.8
```
# cuda 11.3
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements
```

## Run
Only supports running on CONLL03 (`conll`) and MIT restaurant (`mit`) datasets for now. Datasets are under `data` folder.
See `utils/arguments.py` for detailed argument instructions.

### Train
Train and test 
```bash
python train.py --dataset mit --model-name "deepset/bert-large-uncased-whole-word-masking-squad2" --n 10 --num-negatives 2
```

```bash
python train.py --dataset conll --model-name "deepset/bert-large-uncased-whole-word-masking-squad2" --n 10 --num-negatives 0
```

### Inference
Simply test/inference with a specified model
```bash
python inference.py --dataset mit --model-name "deepset/bert-large-uncased-whole-word-masking-squad2"
```