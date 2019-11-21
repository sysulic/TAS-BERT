# TAS-BERT

Target-Aspect-Sentiment Joint Detection for Aspect-Based Sentiment Analysis.

## Requirements

- pytorch: 1.0.1
- python: 3.6.8
- tensorflow: 1.13.1 (only for creating BERT-pytorch-model)
- pytorch-crf: 0.7.2
- numpy: 1.16.4
- nltk: 3.4.4
- sklearn: 0.21.2

## Data Preprocessing

- Download [uncased BERT-Based model](https://github.com/google-research/bert), and run `convert_tf_checkpoint_to_pytorch.py` to create BERT-pytorch-model.
- run `data/data_preprocessing_for_TAS.py` to get preprocessed data.


## Code Structure

- `ABSA_joint.py`: Program Runner.
- `modeling.py`: Program Models.
- `optimization.py`: Optimization for model.
- `processor.py`: Data Processor.
- `tokenization.py`: Tokenization, including three unknown-word-solutions.
- `evaluation_for_ASD.py`, `evaluation_for_TASD_joint.py` and `evaluation_for_TSD.py`: evaluation for ASD, TASD and TSD tasks.
- `evaluation_gold_for_AD_TD_TAD/`: The official evaluation for AD, TD and TAD tasks.
- `ABSA_joint_split.py`, `modeling_split.py` and `evaluation_for_loss_split.py`: Separate detection for Ablation Study.
