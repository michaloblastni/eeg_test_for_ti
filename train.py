from __future__ import annotations

# python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
# python train.py --epochs 100 --repeats 5 --batch-size 128

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 128
NUM_CHANNELS = 16
RECORDING_SECONDS = 60
SAMPLES_PER_SUBJECT = SAMPLE_RATE * RECORDING_SECONDS

LABEL_HEALTHY = 0
LABEL_TARGETED = 1
LABEL_NAMES = {
    LABEL_HEALTHY: "healthy_control",
    LABEL_TARGETED: "targeted",
}

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label: int
    signal: np.ndarray


@dataclass(frozen=True)
class SubjectInferenceBatch:
    x_windows: np.ndarray
    y_subjects: np.ndarray
    subject_index_per_window: np.ndarray


@dataclass(frozen=True)
class PaperConfig:
    data_dir: str = "."
    norm_dir: str = "norm"
    sch_dir: str = "sch"
    epochs: int = 100
    repeats: int = 5
    batch_size: int = 128
    n_splits: int = 10
    seed: int = 42
    train_window_seconds: float = 3.0
    train_stride_seconds: float = 0.125
    test_window_seconds: float = 3.0
    majority_window_seconds: float = 6.0
    majority_vote_stride_seconds: float = 1.5
    normalize_channels: bool = False
    dropout_rate: float = 0.0
    l2_weight: float = 0.0
    early_stopping_patience: int = 20
    output_root: str = "runs"
    max_folds: int = 10
    quick_estimate: bool = False

    @property
    def train_window_samples(self) -> int:
        return int(round(self.train_window_seconds * SAMPLE_RATE))

    @property
    def train_stride_samples(self) -> int:
        return int(round(self.train_stride_seconds * SAMPLE_RATE))

    @property
    def test_window_samples(self) -> int:
        return int(round(self.test_window_seconds * SAMPLE_RATE))

    @property
    def majority_window_samples(self) -> int:
        return int(round(self.majority_window_seconds * SAMPLE_RATE))

    @property
    def majority_vote_stride_samples(self) -> int:
        return int(round(self.majority_vote_stride_seconds * SAMPLE_RATE))


class SlidingWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[SubjectRecord],
        window_size: int,
        stride: int,
    ) -> None:
        if not records:
            raise ValueError("SlidingWindowDataset requires at least one subject.")
        if window_size > SAMPLES_PER_SUBJECT:
            raise ValueError("Window size cannot exceed the subject recording length.")
        if stride <= 0:
            raise ValueError("Stride must be positive.")

        self.records = records
        self.window_size = window_size
        self.stride = stride
        self.window_refs = self._build_window_refs()

    def _build_window_refs(self) -> np.ndarray:
        refs: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            max_start = record.signal.shape[1] - self.window_size
            for start in range(0, max_start + 1, self.stride):
                refs.append((record_index, start))
        return np.asarray(refs, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.window_refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index, sample_start = self.window_refs[index]
        record = self.records[int(record_index)]
        sample_stop = int(sample_start + self.window_size)
        window = record.signal[:, int(sample_start):sample_stop]
        x = torch.from_numpy(window).unsqueeze(0)
        y = torch.tensor(record.label, dtype=torch.long)
        return x, y


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        x = self.conv(x)
        return self.activation(x)


class PaperModel(nn.Module):
    def __init__(self, window_size: int, config: PaperConfig) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, 16, (1, 11), (0, 5)),
            ConvBlock(16, 16, (NUM_CHANNELS, 1), (0, 0)),
            ConvBlock(16, 16, (1, 3), (0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ConvBlock(16, 12, (1, 3), (0, 1)),
            ConvBlock(12, 12, (1, 3), (0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ConvBlock(12, 8, (1, 3), (0, 1)),
            ConvBlock(8, 8, (1, 3), (0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ConvBlock(8, 4, (1, 3), (0, 1)),
            ConvBlock(4, 4, (1, 3), (0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, 1, NUM_CHANNELS, window_size), dtype=torch.float32)
            flattened_features = int(self.feature_extractor(dummy).reshape(1, -1).shape[1])

        self.dropout_before_fc1 = (
            nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()
        )
        self.fc1 = nn.Linear(flattened_features, 4)
        self.dropout_before_fc2 = (
            nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()
        )
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout_before_fc1(x)
        x = self.fc1(x)
        x = self.dropout_before_fc2(x)
        return self.fc2(x)


class SubjectMetricCheckpoint:
    def __init__(
        self,
        validation_batch: SubjectInferenceBatch,
        checkpoint_path: Path,
        patience: int,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self.validation_batch = validation_batch
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.device = device
        self.batch_size = batch_size
        self.best_accuracy = -1.0
        self.best_balanced_accuracy = -1.0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_metrics: dict[str, float | int] | None = None
        self.wait = 0

    def update(
        self,
        model: PaperModel,
        epoch: int,
        logs: dict[str, float],
    ) -> bool:
        metrics = evaluate_subject_inference_batch(
            model=model,
            batch=self.validation_batch,
            device=self.device,
            batch_size=self.batch_size,
        )
        logs["val_subject_accuracy"] = float(metrics["accuracy"])
        logs["val_subject_sensitivity"] = float(metrics["sensitivity"])
        logs["val_subject_specificity"] = float(metrics["specificity"])
        logs["val_subject_balanced_accuracy"] = float(metrics["balanced_accuracy"])
        logs["val_subject_loss"] = float(metrics["subject_loss"])

        current_accuracy = float(metrics["accuracy"])
        current_balanced_accuracy = float(metrics["balanced_accuracy"])
        current_loss = float(metrics["subject_loss"])
        improved = (
            current_balanced_accuracy > self.best_balanced_accuracy
            or (
                math.isclose(
                    current_balanced_accuracy,
                    self.best_balanced_accuracy,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                and current_loss < self.best_loss - 1e-8
            )
        )

        if improved:
            self.best_accuracy = current_accuracy
            self.best_balanced_accuracy = current_balanced_accuracy
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            self.best_metrics = metrics
            self.wait = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False

        self.wait += 1
        return self.patience > 0 and self.wait >= self.patience


def parse_args() -> PaperConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the EEG paper "
            "(Applied Sciences 2024, 14(12), 5048) in PyTorch."
        )
    )
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--norm-dir", type=str, default="norm")
    parser.add_argument("--sch-dir", type=str, default="sch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-window-seconds", type=float, default=3.0)
    parser.add_argument("--train-stride-seconds", type=float, default=0.125)
    parser.add_argument("--test-window-seconds", type=float, default=3.0)
    parser.add_argument("--majority-window-seconds", type=float, default=6.0)
    parser.add_argument("--majority-vote-stride-seconds", type=float, default=1.5)
    parser.add_argument(
        "--normalize-channels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--l2-weight", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument(
        "--quick-estimate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.quick_estimate:
        args.epochs = min(args.epochs, 30)
        args.repeats = min(args.repeats, 2)
        args.max_folds = min(args.max_folds, 2)
        args.early_stopping_patience = min(args.early_stopping_patience, 10)
    return PaperConfig(**vars(args))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_signal_per_channel(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (signal - mean) / std


def load_eeg_file(path: Path, normalize_channels: bool) -> np.ndarray:
    signal = np.loadtxt(path, dtype=np.float32)
    expected_size = NUM_CHANNELS * SAMPLES_PER_SUBJECT
    if signal.size != expected_size:
        raise ValueError(f"{path} has {signal.size} samples; expected {expected_size}.")
    signal = signal.reshape(NUM_CHANNELS, SAMPLES_PER_SUBJECT)
    if normalize_channels:
        signal = normalize_signal_per_channel(signal).astype(np.float32)
    return signal


def load_dataset(config: PaperConfig) -> list[SubjectRecord]:
    data_dir = Path(config.data_dir)
    norm_paths = sorted((data_dir / config.norm_dir).glob("*.eea"))
    sch_paths = sorted((data_dir / config.sch_dir).glob("*.eea"))

    if len(norm_paths) != 39:
        raise ValueError(
            f"Expected 39 healthy-control files, found {len(norm_paths)} in {config.norm_dir}."
        )
    if len(sch_paths) != 45:
        raise ValueError(
            f"Expected 45 targeted files, found {len(sch_paths)} in {config.sch_dir}."
        )

    dataset: list[SubjectRecord] = []
    for path in norm_paths:
        dataset.append(
            SubjectRecord(
                subject_id=path.stem,
                label=LABEL_HEALTHY,
                signal=load_eeg_file(path, normalize_channels=config.normalize_channels),
            )
        )
    for path in sch_paths:
        dataset.append(
            SubjectRecord(
                subject_id=path.stem,
                label=LABEL_TARGETED,
                signal=load_eeg_file(path, normalize_channels=config.normalize_channels),
            )
        )
    return dataset


def create_dataloader(
    records: list[SubjectRecord],
    window_size: int,
    stride: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = SlidingWindowDataset(records=records, window_size=window_size, stride=stride)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_HEALTHY, LABEL_TARGETED])
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_non_overlapping_windows(
    records: list[SubjectRecord],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        for start in range(0, record.signal.shape[1] - window_size + 1, window_size):
            stop = start + window_size
            windows.append(record.signal[:, start:stop])
            labels.append(record.label)
    x = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def make_majority_vote_windows(
    records: list[SubjectRecord],
    majority_window_size: int,
    vote_window_size: int,
    vote_stride: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    vote_windows: list[np.ndarray] = []
    labels: list[int] = []
    votes_per_window = ((majority_window_size - vote_window_size) // vote_stride) + 1
    if votes_per_window < 1:
        raise ValueError("Majority-vote setup must generate at least one vote window.")

    for record in records:
        for start in range(
            0,
            record.signal.shape[1] - majority_window_size + 1,
            majority_window_size,
        ):
            six_second_window = record.signal[:, start:start + majority_window_size]
            labels.append(record.label)
            for vote_start in range(
                0,
                majority_window_size - vote_window_size + 1,
                vote_stride,
            ):
                vote_stop = vote_start + vote_window_size
                vote_windows.append(six_second_window[:, vote_start:vote_stop])

    x = np.stack(vote_windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return x, y, votes_per_window


def majority_vote_predictions(
    vote_predictions: np.ndarray,
    votes_per_window: int,
) -> np.ndarray:
    vote_labels = vote_predictions.argmax(axis=1)
    grouped_votes = vote_labels.reshape(-1, votes_per_window)
    return np.asarray(
        [np.bincount(group, minlength=2).argmax() for group in grouped_votes],
        dtype=np.int32,
    )


def format_score(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fold_summary_line(prefix: str, metrics: dict[str, float | int]) -> str:
    return (
        f"{prefix} acc={format_score(float(metrics['accuracy']))}, "
        f"sens={format_score(float(metrics['sensitivity']))}, "
        f"spec={format_score(float(metrics['specificity']))}"
    )


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def count_non_trainable_parameters(model: nn.Module) -> int:
    return int(sum(buffer.numel() for buffer in model.buffers() if torch.is_floating_point(buffer)))


def compute_probabilistic_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    indices = np.arange(len(y_true), dtype=np.int32)
    return float(np.mean(-np.log(y_prob[indices, y_true])))


def aggregate_subject_probabilities(
    window_predictions: np.ndarray,
    batch: SubjectInferenceBatch,
) -> np.ndarray:
    n_subjects = len(batch.y_subjects)
    aggregated_probabilities = np.zeros((n_subjects, 2), dtype=np.float32)
    window_counts = np.zeros((n_subjects, 1), dtype=np.float32)
    np.add.at(
        aggregated_probabilities,
        batch.subject_index_per_window,
        window_predictions,
    )
    np.add.at(window_counts[:, 0], batch.subject_index_per_window, 1.0)
    aggregated_probabilities /= window_counts
    return aggregated_probabilities


def build_subject_inference_batch(
    records: list[SubjectRecord],
    config: PaperConfig,
) -> SubjectInferenceBatch:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    subject_index_per_window: list[int] = []

    for subject_index, record in enumerate(records):
        labels.append(record.label)
        for start in range(
            0,
            record.signal.shape[1] - config.majority_window_samples + 1,
            config.majority_window_samples,
        ):
            six_second_window = record.signal[:, start:start + config.majority_window_samples]
            for vote_start in range(
                0,
                config.majority_window_samples - config.test_window_samples + 1,
                config.majority_vote_stride_samples,
            ):
                vote_stop = vote_start + config.test_window_samples
                windows.append(six_second_window[:, vote_start:vote_stop])
                subject_index_per_window.append(subject_index)

    return SubjectInferenceBatch(
        x_windows=np.stack(windows, axis=0).astype(np.float32),
        y_subjects=np.asarray(labels, dtype=np.int32),
        subject_index_per_window=np.asarray(subject_index_per_window, dtype=np.int32),
    )


def predict_probabilities(
    model: PaperModel,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            batch_x = torch.from_numpy(x[start:stop]).unsqueeze(1).to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            logits = model(batch_x)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def predict_logits(
    model: PaperModel,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            batch_x = torch.from_numpy(x[start:stop]).unsqueeze(1).to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            logits = model(batch_x)
            out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_subject_inference_batch(
    model: PaperModel,
    batch: SubjectInferenceBatch,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    predictions = predict_probabilities(
        model=model,
        x=batch.x_windows,
        device=device,
        batch_size=batch_size,
    )
    n_subjects = len(batch.y_subjects)
    aggregated_probabilities = aggregate_subject_probabilities(predictions, batch)
    predicted_labels = aggregated_probabilities.argmax(axis=1)
    metrics = compute_metrics(batch.y_subjects, predicted_labels)
    metrics["subjects"] = int(n_subjects)
    metrics["subject_loss"] = compute_probabilistic_loss(
        batch.y_subjects,
        aggregated_probabilities,
    )
    return metrics


def evaluate_best_model(
    model: PaperModel,
    test_records: list[SubjectRecord],
    config: PaperConfig,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, float | int], dict[str, float | int]]:
    x_plain, y_plain = make_non_overlapping_windows(
        test_records,
        window_size=config.test_window_samples,
    )
    plain_predictions = predict_probabilities(
        model=model,
        x=x_plain,
        device=device,
        batch_size=config.batch_size,
    )
    plain_metrics = compute_metrics(y_plain, plain_predictions.argmax(axis=1))

    x_mv, y_mv, votes_per_window = make_majority_vote_windows(
        test_records,
        majority_window_size=config.majority_window_samples,
        vote_window_size=config.test_window_samples,
        vote_stride=config.majority_vote_stride_samples,
    )
    mv_predictions = predict_probabilities(
        model=model,
        x=x_mv,
        device=device,
        batch_size=config.batch_size,
    )
    mv_metrics = compute_metrics(
        y_mv,
        majority_vote_predictions(mv_predictions, votes_per_window=votes_per_window),
    )

    subject_metrics = evaluate_subject_inference_batch(
        model=model,
        batch=build_subject_inference_batch(test_records, config),
        device=device,
        batch_size=config.batch_size,
    )
    return plain_metrics, mv_metrics, subject_metrics


def load_checkpoint(
    model: PaperModel,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def average_predictions_from_checkpoints(
    checkpoint_paths: list[Path],
    config: PaperConfig,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required for ensemble evaluation.")

    model = PaperModel(window_size=config.train_window_samples, config=config).to(device)
    probability_sum = np.zeros((x.shape[0], 2), dtype=np.float32)

    for checkpoint_path in checkpoint_paths:
        load_checkpoint(model, checkpoint_path, device)
        probability_sum += predict_probabilities(
            model=model,
            x=x,
            device=device,
            batch_size=config.batch_size,
        )

    return probability_sum / float(len(checkpoint_paths))


def average_logits_from_checkpoints(
    checkpoint_paths: list[Path],
    config: PaperConfig,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required for ensemble evaluation.")

    model = PaperModel(window_size=config.train_window_samples, config=config).to(device)
    logit_sum = np.zeros((x.shape[0], 2), dtype=np.float32)

    for checkpoint_path in checkpoint_paths:
        load_checkpoint(model, checkpoint_path, device)
        logit_sum += predict_logits(
            model=model,
            x=x,
            device=device,
            batch_size=config.batch_size,
        )

    return logit_sum / float(len(checkpoint_paths))


def evaluate_checkpoint_ensemble(
    checkpoint_paths: list[Path],
    records: list[SubjectRecord],
    config: PaperConfig,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, float | int], dict[str, float | int]]:
    x_plain, y_plain = make_non_overlapping_windows(
        records,
        window_size=config.test_window_samples,
    )
    plain_predictions = average_predictions_from_checkpoints(
        checkpoint_paths,
        config,
        x_plain,
        device,
    )
    plain_metrics = compute_metrics(y_plain, plain_predictions.argmax(axis=1))

    x_mv, y_mv, votes_per_window = make_majority_vote_windows(
        records,
        majority_window_size=config.majority_window_samples,
        vote_window_size=config.test_window_samples,
        vote_stride=config.majority_vote_stride_samples,
    )
    mv_predictions = average_predictions_from_checkpoints(
        checkpoint_paths,
        config,
        x_mv,
        device,
    )
    mv_metrics = compute_metrics(
        y_mv,
        majority_vote_predictions(mv_predictions, votes_per_window=votes_per_window),
    )

    subject_batch = build_subject_inference_batch(records, config)
    subject_window_predictions = average_predictions_from_checkpoints(
        checkpoint_paths,
        config,
        subject_batch.x_windows,
        device,
    )
    aggregated_subject_probabilities = aggregate_subject_probabilities(
        subject_window_predictions,
        subject_batch,
    )
    subject_metrics = compute_metrics(
        subject_batch.y_subjects,
        aggregated_subject_probabilities.argmax(axis=1),
    )
    subject_metrics["subjects"] = int(len(subject_batch.y_subjects))
    subject_metrics["subject_loss"] = compute_probabilistic_loss(
        subject_batch.y_subjects,
        aggregated_subject_probabilities,
    )
    return plain_metrics, mv_metrics, subject_metrics


def build_folds(labels: np.ndarray, config: PaperConfig) -> list[np.ndarray]:
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    folds: list[np.ndarray] = []
    dummy_x = np.zeros((len(labels), 1), dtype=np.float32)
    for _, test_indices in splitter.split(dummy_x, labels):
        folds.append(test_indices)
    return folds


def describe_split(name: str, records: Iterable[SubjectRecord]) -> str:
    items = list(records)
    healthy = sum(record.label == LABEL_HEALTHY for record in items)
    targeted = sum(record.label == LABEL_TARGETED for record in items)
    return (
        f"{name}: total={len(items)}, "
        f"healthy={healthy}, targeted={targeted}"
    )


def train_one_epoch(
    model: PaperModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        batch_y = batch_y.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = int(batch_y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def evaluate_window_loader(
    model: PaperModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            batch_y = batch_y.to(
                device=device,
                dtype=torch.long,
                non_blocking=device.type == "cuda",
            )
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            batch_size = int(batch_y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_repeat(
    model: PaperModel,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_subject_batch: SubjectInferenceBatch,
    config: PaperConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, list[float]], SubjectMetricCheckpoint]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=config.l2_weight,
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint = SubjectMetricCheckpoint(
        validation_batch=val_subject_batch,
        checkpoint_path=checkpoint_path,
        patience=config.early_stopping_patience,
        device=device,
        batch_size=config.batch_size,
    )

    history: dict[str, list[float]] = {
        "accuracy": [],
        "loss": [],
        "val_accuracy": [],
        "val_loss": [],
        "val_subject_accuracy": [],
        "val_subject_sensitivity": [],
        "val_subject_specificity": [],
        "val_subject_balanced_accuracy": [],
        "val_subject_loss": [],
    }

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss, val_accuracy = evaluate_window_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        logs = {
            "accuracy": float(train_accuracy),
            "loss": float(train_loss),
            "val_accuracy": float(val_accuracy),
            "val_loss": float(val_loss),
        }

        should_stop = checkpoint.update(model=model, epoch=epoch, logs=logs)

        for key, value in logs.items():
            history.setdefault(key, []).append(float(value))

        elapsed_seconds = time.perf_counter() - epoch_start
        ms_per_step = (elapsed_seconds * 1000.0) / max(1, len(train_loader))

        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(
            f"{len(train_loader)}/{len(train_loader)} - "
            f"{elapsed_seconds:.0f}s - {ms_per_step:.0f}ms/step - "
            f"accuracy: {logs['accuracy']:.4f} - "
            f"loss: {logs['loss']:.4f} - "
            f"val_accuracy: {logs['val_accuracy']:.4f} - "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_subject_accuracy: {logs['val_subject_accuracy']:.4f} - "
            f"val_subject_sensitivity: {logs['val_subject_sensitivity']:.4f} - "
            f"val_subject_specificity: {logs['val_subject_specificity']:.4f} - "
            f"val_subject_balanced_accuracy: {logs['val_subject_balanced_accuracy']:.4f} - "
            f"val_subject_loss: {logs['val_subject_loss']:.4f}"
        )

        if should_stop:
            break

    return history, checkpoint


def run_experiment(config: PaperConfig) -> None:
    dataset = load_dataset(config)
    labels = np.asarray([record.label for record in dataset], dtype=np.int32)
    all_indices = np.arange(len(dataset), dtype=np.int32)
    folds = build_folds(labels, config)
    device = get_device()

    if config.max_folds < 1 or config.max_folds > config.n_splits:
        raise ValueError("--max-folds must be between 1 and n_splits.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_root) / f"paper_reproduction_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(run_dir / "config.json", asdict(config))

    repeat_rows: list[dict[str, float | int | str]] = []
    fold_rows: list[dict[str, float | int | str]] = []
    best_majority_scores: list[float] = []
    best_plain_scores: list[float] = []
    best_subject_scores: list[float] = []
    model_trainable_param_count: int | None = None
    model_total_param_count: int | None = None

    print(f"Loaded {len(dataset)} subjects from {Path(config.data_dir).resolve()}")
    print(f"Paper training windows: {config.train_window_samples} samples")
    print(f"Paper training stride: {config.train_stride_samples} samples")
    print(f"Paper majority-vote stride: {config.majority_vote_stride_samples} samples")
    print(
        "Preprocessing/config: "
        f"normalize_channels={config.normalize_channels}, "
        f"dropout_rate={config.dropout_rate}, "
        f"l2_weight={config.l2_weight}, "
        f"early_stopping_patience={config.early_stopping_patience}"
    )
    if config.quick_estimate:
        print(
            "Quick estimate mode: "
            f"epochs={config.epochs}, repeats={config.repeats}, max_folds={config.max_folds}"
        )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    for fold_index in range(config.max_folds):
        test_indices = folds[fold_index]
        val_indices = folds[(fold_index + 1) % config.n_splits]
        train_indices = np.setdiff1d(all_indices, np.concatenate([test_indices, val_indices]))

        train_records = [dataset[i] for i in train_indices]
        val_records = [dataset[i] for i in val_indices]
        test_records = [dataset[i] for i in test_indices]
        val_subject_batch = build_subject_inference_batch(val_records, config)

        print()
        print(f"Fold {fold_index + 1}/{config.max_folds}")
        print(describe_split("Train", train_records))
        print(describe_split("Validation", val_records))
        print(describe_split("Test", test_records))

        best_repeat_row: dict[str, float | int | str] | None = None
        best_repeat_checkpoint: Path | None = None
        fold_checkpoint_paths: list[Path] = []

        for repeat_index in range(config.repeats):
            repeat_seed = config.seed + (fold_index * 100) + repeat_index
            set_global_seed(repeat_seed)

            print(f"Repeat {repeat_index + 1}/{config.repeats}")

            train_loader = create_dataloader(
                records=train_records,
                window_size=config.train_window_samples,
                stride=config.train_stride_samples,
                batch_size=config.batch_size,
                shuffle=True,
                seed=repeat_seed,
                device=device,
            )
            val_loader = create_dataloader(
                records=val_records,
                window_size=config.train_window_samples,
                stride=config.train_stride_samples,
                batch_size=config.batch_size,
                shuffle=False,
                seed=repeat_seed,
                device=device,
            )

            model = PaperModel(window_size=config.train_window_samples, config=config).to(device)

            if model_trainable_param_count is None or model_total_param_count is None:
                model_trainable_param_count = count_trainable_parameters(model)
                model_total_param_count = (
                    model_trainable_param_count + count_non_trainable_parameters(model)
                )
                print(
                    "Model parameter count: "
                    f"trainable={model_trainable_param_count}, "
                    f"total={model_total_param_count}"
                )

            checkpoint_path = (
                run_dir / f"fold_{fold_index + 1:02d}_repeat_{repeat_index + 1:02d}.pt"
            )

            history, checkpoint = train_repeat(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_subject_batch=val_subject_batch,
                config=config,
                checkpoint_path=checkpoint_path,
                device=device,
            )

            load_checkpoint(model, checkpoint_path, device)
            fold_checkpoint_paths.append(checkpoint_path)

            best_epoch_index = min(
                max(0, int(checkpoint.best_epoch) - 1),
                max(0, len(history["val_accuracy"]) - 1),
            )
            best_val_accuracy = float(history["val_accuracy"][best_epoch_index])
            best_val_subject_accuracy = float(checkpoint.best_accuracy)
            best_val_subject_balanced_accuracy = float(checkpoint.best_balanced_accuracy)
            best_val_subject_loss = float(checkpoint.best_loss)

            plain_metrics, mv_metrics, subject_metrics = evaluate_best_model(
                model=model,
                test_records=test_records,
                config=config,
                device=device,
            )

            history_path = (
                run_dir / f"fold_{fold_index + 1:02d}_repeat_{repeat_index + 1:02d}_history.json"
            )
            save_json(history_path, history)

            row: dict[str, float | int | str] = {
                "fold": fold_index + 1,
                "repeat": repeat_index + 1,
                "seed": repeat_seed,
                "val_accuracy": best_val_accuracy,
                "val_subject_accuracy": best_val_subject_accuracy,
                "val_subject_balanced_accuracy": best_val_subject_balanced_accuracy,
                "val_subject_loss": best_val_subject_loss,
                "best_epoch": int(checkpoint.best_epoch),
                "plain_accuracy": float(plain_metrics["accuracy"]),
                "plain_sensitivity": float(plain_metrics["sensitivity"]),
                "plain_specificity": float(plain_metrics["specificity"]),
                "mv_accuracy": float(mv_metrics["accuracy"]),
                "mv_sensitivity": float(mv_metrics["sensitivity"]),
                "mv_specificity": float(mv_metrics["specificity"]),
                "subject_accuracy": float(subject_metrics["accuracy"]),
                "subject_sensitivity": float(subject_metrics["sensitivity"]),
                "subject_specificity": float(subject_metrics["specificity"]),
                "checkpoint": checkpoint_path.name,
            }
            repeat_rows.append(row)

            print(
                f" Repeat {repeat_index + 1}/{config.repeats}: "
                f"val_acc={format_score(best_val_accuracy)} | "
                f"val_subject_acc={format_score(best_val_subject_accuracy)} | "
                f"val_subject_loss={best_val_subject_loss:.4f} | "
                f"{fold_summary_line('plain', plain_metrics)} | "
                f"{fold_summary_line('mv', mv_metrics)} | "
                f"{fold_summary_line('subject', subject_metrics)}"
            )

            if (
                best_repeat_row is None
                or best_val_subject_balanced_accuracy > float(best_repeat_row["val_subject_balanced_accuracy"])
                or (
                    math.isclose(
                        best_val_subject_balanced_accuracy,
                        float(best_repeat_row["val_subject_balanced_accuracy"]),
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                    and best_val_subject_loss < float(best_repeat_row["val_subject_loss"]) - 1e-8
                )
            ):
                best_repeat_row = row
                best_repeat_checkpoint = checkpoint_path

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if best_repeat_row is None or best_repeat_checkpoint is None:
            raise RuntimeError("A fold finished without a selected repeat.")

        ensemble_plain_metrics, ensemble_mv_metrics, ensemble_subject_metrics = (
            evaluate_checkpoint_ensemble(
                checkpoint_paths=fold_checkpoint_paths,
                records=test_records,
                config=config,
                device=device,
            )
        )
        _, _, ensemble_val_subject_metrics = evaluate_checkpoint_ensemble(
            checkpoint_paths=fold_checkpoint_paths,
            records=val_records,
            config=config,
            device=device,
        )

        ensemble_row: dict[str, float | int | str] = {
            "fold": fold_index + 1,
            "repeat": "ensemble",
            "seed": "ensemble",
            "val_accuracy": float("nan"),
            "val_subject_accuracy": float(ensemble_val_subject_metrics["accuracy"]),
            "val_subject_balanced_accuracy": float(
                ensemble_val_subject_metrics["balanced_accuracy"]
            ),
            "val_subject_loss": float(ensemble_val_subject_metrics["subject_loss"]),
            "best_epoch": "",
            "plain_accuracy": float(ensemble_plain_metrics["accuracy"]),
            "plain_sensitivity": float(ensemble_plain_metrics["sensitivity"]),
            "plain_specificity": float(ensemble_plain_metrics["specificity"]),
            "mv_accuracy": float(ensemble_mv_metrics["accuracy"]),
            "mv_sensitivity": float(ensemble_mv_metrics["sensitivity"]),
            "mv_specificity": float(ensemble_mv_metrics["specificity"]),
            "subject_accuracy": float(ensemble_subject_metrics["accuracy"]),
            "subject_sensitivity": float(ensemble_subject_metrics["sensitivity"]),
            "subject_specificity": float(ensemble_subject_metrics["specificity"]),
            "checkpoint": f"{len(fold_checkpoint_paths)}-repeat-ensemble",
        }

        fold_rows.append(ensemble_row)
        best_plain_scores.append(float(ensemble_row["plain_accuracy"]))
        best_majority_scores.append(float(ensemble_row["mv_accuracy"]))
        best_subject_scores.append(float(ensemble_row["subject_accuracy"]))

        print(
            " Best single repeat by validation subject metrics: "
            f"{int(best_repeat_row['repeat'])} "
            f"(val_subject_acc={format_score(float(best_repeat_row['val_subject_accuracy']))}, "
            f"subject_acc={format_score(float(best_repeat_row['subject_accuracy']))})"
        )
        print(
            " Repeat ensemble: "
            f"{fold_summary_line('plain', ensemble_plain_metrics)} | "
            f"{fold_summary_line('mv', ensemble_mv_metrics)} | "
            f"{fold_summary_line('subject', ensemble_subject_metrics)}"
        )
        print(
            " Running mean after fold: "
            f"plain={format_score(float(np.mean(best_plain_scores)))} | "
            f"mv={format_score(float(np.mean(best_majority_scores)))} | "
            f"subject={format_score(float(np.mean(best_subject_scores)))}"
        )

    repeat_csv = run_dir / "repeat_results.csv"
    with repeat_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(repeat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(repeat_rows)

    fold_csv = run_dir / "fold_best_results.csv"
    with fold_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    summary = {
        "framework": "pytorch",
        "model_trainable_parameter_count": model_trainable_param_count,
        "model_total_parameter_count": model_total_param_count,
        "paper_model_parameter_count": 7358,
        "mean_plain_accuracy": float(np.mean(best_plain_scores)),
        "mean_majority_accuracy": float(np.mean(best_majority_scores)),
        "mean_majority_accuracy_percent": float(np.mean(best_majority_scores) * 100.0),
        "mean_subject_accuracy": float(np.mean(best_subject_scores)),
        "mean_subject_accuracy_percent": float(np.mean(best_subject_scores) * 100.0),        
        "completed_folds": config.max_folds,
        "completed_repeats_per_fold": config.repeats,
        "notes": [
            "This script keeps the paper-style crops while reporting both segment-level and full-subject metrics.",
            "Whole-subject checkpoint selection and repeat ensembling are enabled by default.",
            "Channel-wise normalization, dropout, and L2 regularization are available as optional flags but are disabled by default.",
            "The paper does not report a batch size or a random seed; batch size is exposed as a CLI argument and defaults to 128.",
        ],
    }

    save_json(run_dir / "summary.json", summary)

    print()
    print(f"Run artifacts saved to {run_dir.resolve()}")
    print(f"Mean plain accuracy: {format_score(float(summary['mean_plain_accuracy']))}")
    print(f"Mean majority-vote accuracy: {format_score(float(summary['mean_majority_accuracy']))}")
    print(f"Mean subject accuracy: {format_score(float(summary['mean_subject_accuracy']))}")


if __name__ == "__main__":
    run_experiment(parse_args())
