# EEG-Based Scientific Test for Targeted Individuals v 0.0.1

This workspace contains a PyTorch implementation based on the paper "Schizophrenia Detection on EEG Signals Using an Ensemble of a Lightweight Convolutional Neural Network" ([MDPI](https://www.mdpi.com/2076-3417/14/12/5048)).

It is implemented as a regular Python script rather than a notebook because the workflow is a full repeated cross-validation pipeline with checkpoints and saved fold results.

The current defaults stay close to the stronger raw-signal baseline, while keeping better repeat selection and saved subject-level metrics.


This is a research prototype. The dataset used is good enough to train an AI model, however a new dataset is needed. Can targeted individuals anonymously record their own EEG, i.e. using OpenBCI 32bit 8ch and contribute their data to this project?

Once 40 TIs have contributed, and 40 controls who are not targeted, the current code should be able to classify each subject with a high accuracy.
On the sample dataset, mean subject accuracy is 90.42%.

More about the OpenBCI hardware is at ([TargetedIndividSci](https://www.reddit.com/r/TargetedIndividSci/comments/1mm3s4c/openbci_32bit_8_channels_at_a_low_cost/)).

If this experiment is successful, targeted individuals will have a scientific test.

Progress: 0 TIs have contributed. 40 are needed.

# EEG-Based Scientific Test for Whether a Subject is a Targeted Individual

Download the sample dataset from ([Brain Bio MSU](http://brain.bio.msu.ru/eeg_schizophrenia.htm)).


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

## Record your EEG
In order to contribute to this project, use OpenBCI 32bit 8ch with the UltraCortex Mark III or IV. 

Do not use the default OpenBCI 8-channel map. Measure these 8 channels: F7, F3, F4, F8, T7(T3), T8(T4), P7(T5), P8(T6).

Record your EEG for 60s. Avoid talking, jaw clenching, blinking bursts, and head movement.

Then, you can anonymously send your EEG to michaloblastni(at)gmail.com and state whether you self-identify as targeted or healthy control. 
In order to self-identify as targeted, you will need to confirm you hear external auditory intrusions that happen in response to what you think, and that often 
reply to what you were thinking.

Your EEG will be added to the training dataset, and used for training and testing an AI-based classifier.

## Support For Using a Trained Model is Already There
Once the model is trained, it is possibly to run classify.py similar to the following:

```bash
$ python3 classify.py \
  --checkpoint runs/paper_reproduction_20260321_202239/fold_01_repeat_01.pt \
  --checkpoint runs/paper_reproduction_20260321_202239/fold_01_repeat_02.pt \
  test_subject.eea
```

[Introductory article](https://www.reddit.com/r/TargetedIndividSci/comments/1s1gz04/eegbased_scientific_test_for_whether_someone_is_a/)
  
It will classify using the new AI model whether the test subject is a targeted individual or a healthy control, and it will also report the probability value P.

ZUNA can remove artifacts and upsample from 8 to 16 channels.
