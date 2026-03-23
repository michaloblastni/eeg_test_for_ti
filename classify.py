"""
Load a trained checkpoint (.pt) from a paper_reproduction run and classify new .eea files.

Training already saves weights under runs/<run_id>/fold_XX_repeat_YY.pt — that *is* your model file.
This script loads one or more checkpoints (averaging probabilities if you pass several) and prints
class probabilities using the same whole-subject aggregation as evaluation. Use --details for mean
logits, softmax of mean logits, and per-window probability spread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper_reproduction import (
    LABEL_NAMES,
    PaperConfig,
    SubjectRecord,
    aggregate_subject_probabilities,
    average_logits_from_checkpoints,
    average_predictions_from_checkpoints,
    build_subject_inference_batch,
    get_device,
    load_eeg_file,
)


def config_from_json(path: Path) -> PaperConfig:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return PaperConfig(**data)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify .eea recordings using saved PyTorch checkpoints from paper_reproduction.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples (Linux/WSL — use real paths; the folder name is paper_reproduction_<date>):\n"
            "  ls runs/*/fold_*_repeat_*.pt\n"
            "  python3 classify.py --checkpoint runs/paper_reproduction_20260321_210250/fold_01_repeat_01.pt subject_1.eea\n"
            "\n"
            "Bash line continuation uses backslash, not ^ (that is Windows cmd.exe)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.json from the same training run (default: config.json next to the first checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
        required=True,
        help="Path to a .pt file (repeat for multiple checkpoints; probabilities are averaged).",
    )
    parser.add_argument(
        "eea_files",
        type=Path,
        nargs="+",
        help="One or more .eea files (16 channels × 7680 samples).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help=(
            "Print mean logits (checkpoint-averaged, then subject-averaged over windows), "
            "softmax of those logits, and per-window spread of ensemble P(healthy)/P(targeted)."
        ),
    )
    args = parser.parse_args()

    checkpoints = [p.resolve() for p in args.checkpoints]
    for p in checkpoints:
        if not p.is_file():
            hint = ""
            if "..." in str(p):
                hint = (
                    " Replace '...' with your real run directory name "
                    "(see `ls runs/`). Example: runs/paper_reproduction_20260321_210250/fold_01_repeat_01.pt"
                )
            raise FileNotFoundError(f"Checkpoint not found: {p}.{hint}")

    config_path = args.config
    if config_path is None:
        config_path = checkpoints[0].parent / "config.json"
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Need --config {config_path} (copy config.json from the training run folder)."
        )

    config = config_from_json(config_path)
    device = get_device()

    for eea_path in args.eea_files:
        eea_path = eea_path.resolve()
        signal = load_eeg_file(eea_path, normalize_channels=config.normalize_channels)
        record = SubjectRecord(subject_id=eea_path.stem, label=0, signal=signal)
        batch = build_subject_inference_batch([record], config)

        probs = average_predictions_from_checkpoints(
            checkpoints,
            config,
            batch.x_windows,
            device,
        )
        agg = aggregate_subject_probabilities(probs, batch)
        pred = int(agg.argmax(axis=1)[0])
        p0, p1 = float(agg[0, 0]), float(agg[0, 1])

        print(f"{eea_path.name}")
        print(f"  predicted: {LABEL_NAMES[pred]} (class {pred})")
        print(f"  P(healthy)={p0:.4f}  P(targeted)={p1:.4f}")

        if args.details:
            logits_w = average_logits_from_checkpoints(
                checkpoints,
                config,
                batch.x_windows,
                device,
            )
            agg_logits = aggregate_subject_probabilities(logits_w, batch)
            al0, al1 = float(agg_logits[0, 0]), float(agg_logits[0, 1])
            from_logits = softmax_rows(agg_logits)[0]
            print("  details:")
            print(
                f"    mean logit (per window, checkpoint-avg, then subject-mean): "
                f"healthy={al0:.4f}  targeted={al1:.4f}"
            )
            print(
                "    softmax(mean logit): "
                f"P(healthy)={from_logits[0]:.4f}  P(targeted)={from_logits[1]:.4f}"
            )
            ph, ps = probs[:, 0], probs[:, 1]
            n_win = probs.shape[0]
            print(f"    per-window ensemble softmax ({n_win} windows): P(healthy) min/median/max = "
                  f"{float(np.min(ph)):.4f} / {float(np.median(ph)):.4f} / {float(np.max(ph)):.4f}")
            print(f"    per-window ensemble softmax: P(targeted) min/median/max = "
                  f"{float(np.min(ps)):.4f} / {float(np.median(ps)):.4f} / {float(np.max(ps)):.4f}")
            print(f"    per-window P(targeted): std={float(np.std(ps)):.4f}")


if __name__ == "__main__":
    main()
