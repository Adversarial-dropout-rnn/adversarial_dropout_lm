# Adversarial Dropout (Penn Treebank Language Modeling)

## A use-case of adversarial dropout on language modeling task (Penn Treebank, PTB)
(Adversarial dropout for recurrent neural network)

This implementation added our algorithm on the original code from the AWD-LSTM Language Model (The base LSTM structure of our algorithm)
Our approach achieved ``58.4 (validation)`` and ``56.1 (test)`` after fine-tuning

## Software Requirements
python 3 and PyTorch 0.3

## Implementation

Learning the model with the adversarial dropout

```python3 main.py --method=AdD --name=PTB_AdD```

Finetuning the model 

```python3 finetune.py --data=data/penn --save=output/PTB_AdD.pt```

Evaluating the learned model

```python3 eval.py --data=data/penn --model_path=output/PTB_AdD.pt```

