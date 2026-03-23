# EEG-Based Scientific Test for Targeted Individuals
EEG-Based Scientific Test for Whether a Subject is a Targeted Individual

This workspace contains a PyTorch implementation based on the paper "Schizophrenia Detection on EEG Signals Using an Ensemble of a Lightweight Convolutional Neural Network" ([MDPI](https://www.mdpi.com/2076-3417/14/12/5048)).

It is implemented as a regular Python script rather than a notebook because the workflow is a full repeated cross-validation pipeline with checkpoints and saved fold results.

The current defaults stay close to the stronger raw-signal baseline, while keeping better repeat selection and saved subject-level metrics.

## What it keeps from the paper

- Dataset: the 84-subject Moscow State University resting-state EEG dataset described [here](http://brain.bio.msu.ru/eeg_schizophrenia.htm)
- Input data: raw 16-channel EEG signals loaded directly from the `.eea` files
- Training crops: 3 s windows with 0.125 s stride
- Evaluation without majority voting: non-overlapping 3 s windows
- Evaluation with majority voting: non-overlapping 6 s windows, each split into three overlapping 3 s sub-windows with 1.5 s stride
- Cross-validation: subject-level stratified 10-fold setup
- Train/validation/test split per run: 8 folds train, 1 fold validation, 1 fold test
- Backbone model: the paper's pyramidal CNN `16 -> 16 -> 16 -> 12 -> 12 -> 8 -> 8 -> 4 -> 4`, with four max-pooling layers and `FC1=4`, `FC2=2`
- Optimizer: Adam with paper defaults (`lr=0.001`, `beta1=0.9`, `beta2=0.999`)
- Epochs: 100
- Repeated runs: 5 per fold

## Extra options

- Optional channel-wise z-score normalization for each subject
- Optional L2 regularization and dropout for experimentation
- Whole-subject validation and checkpoint selection
- Whole-subject inference by averaging probabilities across the full recording

## Important note

The paper does not state a batch size or random seed. The script exposes both. The default batch size is `128`.

In this dataset, the raw recordings show class-level amplitude offsets, so normalization is available but disabled by default.

## Run

Default run:

```bash
python train.py --epochs 100 --repeats 5 --batch-size 128
```

Quick smoke test:

```bash
python train.py --epochs 1 --repeats 1 --max-folds 1
```

## Outputs

Each run creates a timestamped folder under `runs/` containing:

- `config.json`
- `repeat_results.csv`
- `fold_best_results.csv`
- `summary.json`
- saved PyTorch checkpoints for each repeat
- per-repeat training history JSON files

The summary now includes both the paper-style segment metrics and the more practical subject-level accuracy.

For GPU on WSL, install PyTorch in the Ubuntu environment you use for training and verify `torch.cuda.is_available()` before starting a full run.
