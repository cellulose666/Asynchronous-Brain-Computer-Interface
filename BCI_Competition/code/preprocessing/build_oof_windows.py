# =============================================================================
# Implementation of: Simplified BNCI2014001 OOF Preprocessing
#
# Reference:
#   Project-specific preprocessing for asynchronous BNCI2014001 motor imagery.
#   The pipeline builds clean continuous EEG segments, applies per-segment
#   causal filtering, cuts fixed windows, labels idle/task/MI, and assigns
#   leave-one-run-out OOF folds.
#
# Source: No external code copied.
# =============================================================================
"""Build a simple OOF-ready BNCI2014001 window dataset.

Output:
  BCI_Competition/data/processed/bnci2014001_oof_windows.npz

Arrays:
  X        float32, shape (n_windows, 22, 500)
  y        int64, 0 idle / 1 left_hand / 2 right_hand / 3 feet / 4 tongue
  subject  int64
  session  int64, 0 train / 1 test
  run      int64, run index within session
  fold     int64, train-session validation fold id; -1 for test session
  split    int64, 0 train-session windows / 2 official test-session windows
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
DATA_ROOT = PROJECT_ROOT / "data" / "public" / "BNCI2014001"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"

SAMPLING_RATE = 250
WINDOW_SAMPLES = 500
STRIDE_SAMPLES = 125
LOW_HZ = 8.0
HIGH_HZ = 30.0
FILTER_ORDER = 4

CLASS_TO_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
CLASS_NAMES = ["idle", "left_hand", "right_hand", "feet", "tongue"]


@dataclass(frozen=True)
class TaskEvent:
    start: int
    stop: int
    label: int
    name: str


@dataclass(frozen=True)
class CleanSegment:
    start: int
    stop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["all"], help="subject ids or 'all'")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def configure_data_cache(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(data_root)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(data_root)
    mne.set_config("MNE_DATA", str(data_root), set_env=True)
    mne.set_config("MNE_DATASETS_BNCI_PATH", str(data_root), set_env=True)


def resolve_subjects(dataset, requested: list[str]) -> list[int]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(dataset.subject_list)
    return [int(item) for item in requested]


def eeg_data(raw: mne.io.BaseRaw) -> np.ndarray:
    picked = raw.copy().pick("eeg")
    if int(round(picked.info["sfreq"])) != SAMPLING_RATE:
        raise RuntimeError(f"Expected native {SAMPLING_RATE} Hz, got {picked.info['sfreq']}")
    if len(picked.ch_names) != 22:
        raise RuntimeError(f"Expected 22 EEG channels, got {len(picked.ch_names)}: {picked.ch_names}")
    return picked.get_data().astype(np.float32)


def task_events(raw: mne.io.BaseRaw) -> list[TaskEvent]:
    events: list[TaskEvent] = []
    for onset_s, duration_s, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        name = str(desc)
        if name not in CLASS_TO_ID:
            continue
        start = int(round(float(onset_s) * SAMPLING_RATE))
        stop = int(round((float(onset_s) + float(duration_s)) * SAMPLING_RATE))
        events.append(TaskEvent(start=start, stop=stop, label=CLASS_TO_ID[name], name=name))
    return events


def artifact_intervals(raw: mne.io.BaseRaw) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for onset_s, duration_s, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        name = str(desc).lower()
        if "bad" not in name and "artifact" not in name and "reject" not in name:
            continue
        start = int(round(float(onset_s) * SAMPLING_RATE))
        stop = int(round((float(onset_s) + float(duration_s)) * SAMPLING_RATE))
        if stop > start:
            intervals.append((start, stop))
    return sorted(intervals)


def clean_segments(n_samples: int, artifacts: list[tuple[int, int]]) -> list[CleanSegment]:
    segments: list[CleanSegment] = []
    cursor = 0
    for start, stop in artifacts:
        start = max(0, min(start, n_samples))
        stop = max(0, min(stop, n_samples))
        if start - cursor >= WINDOW_SAMPLES:
            segments.append(CleanSegment(cursor, start))
        cursor = max(cursor, stop)
    if n_samples - cursor >= WINDOW_SAMPLES:
        segments.append(CleanSegment(cursor, n_samples))
    return segments


def causal_filter_segment(segment: np.ndarray) -> np.ndarray:
    sos = butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=SAMPLING_RATE, output="sos")
    return sosfilt(sos, segment, axis=1).astype(np.float32)


def overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> bool:
    return a_start < b_stop and a_stop > b_start


def label_window(start: int, stop: int, events: list[TaskEvent]) -> int | None:
    overlapping = [event for event in events if overlap(start, stop, event.start, event.stop)]
    if not overlapping:
        return 0
    contained = [event for event in overlapping if start >= event.start and stop <= event.stop]
    if len(contained) == 1 and len(overlapping) == 1:
        return contained[0].label
    return None


def build_run_windows(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray, dict]:
    signal = eeg_data(raw)
    events = task_events(raw)
    artifacts = artifact_intervals(raw)
    segments = clean_segments(signal.shape[1], artifacts)

    windows: list[np.ndarray] = []
    labels: list[int] = []
    dropped_boundary = 0

    for segment in segments:
        filtered = causal_filter_segment(signal[:, segment.start : segment.stop])
        for local_start in range(0, filtered.shape[1] - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            global_start = segment.start + local_start
            global_stop = global_start + WINDOW_SAMPLES
            label = label_window(global_start, global_stop, events)
            if label is None:
                dropped_boundary += 1
                continue
            windows.append(filtered[:, local_start : local_start + WINDOW_SAMPLES])
            labels.append(label)

    if not windows:
        return (
            np.empty((0, 22, WINDOW_SAMPLES), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {"segments": len(segments), "task_events": len(events), "artifacts": len(artifacts), "dropped_boundary": dropped_boundary},
        )
    info = {
        "segments": len(segments),
        "task_events": len(events),
        "artifacts": len(artifacts),
        "dropped_boundary": dropped_boundary,
        "windows": len(windows),
    }
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64), info


def build_dataset(subjects: list[int]) -> tuple[dict[str, np.ndarray], list[dict]]:
    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001()
    subject_data = dataset.get_data(subjects=subjects)

    arrays: dict[str, list[np.ndarray]] = {
        "X": [],
        "y": [],
        "subject": [],
        "session": [],
        "run": [],
        "fold": [],
        "split": [],
    }
    records: list[dict] = []

    for subject in subjects:
        for session_name, runs in subject_data[subject].items():
            is_train_session = "train" in session_name.lower()
            session_id = 0 if is_train_session else 1
            split_id = 0 if is_train_session else 2
            for run_index, (run_name, raw) in enumerate(runs.items()):
                X_run, y_run, info = build_run_windows(raw)
                n = len(y_run)
                arrays["X"].append(X_run)
                arrays["y"].append(y_run)
                arrays["subject"].append(np.full(n, subject, dtype=np.int64))
                arrays["session"].append(np.full(n, session_id, dtype=np.int64))
                arrays["run"].append(np.full(n, run_index, dtype=np.int64))
                arrays["fold"].append(np.full(n, run_index if is_train_session else -1, dtype=np.int64))
                arrays["split"].append(np.full(n, split_id, dtype=np.int64))
                record = {
                    "subject": subject,
                    "session": session_name,
                    "run": run_name,
                    "run_index": run_index,
                    "split": "train_session" if is_train_session else "test_session",
                    **info,
                    "label_counts": np.bincount(y_run, minlength=len(CLASS_NAMES)).astype(int).tolist(),
                }
                records.append(record)
                print(f"s{subject:02d} {session_name}/{run_name}: windows={n} labels={record['label_counts']}")

    output = {
        key: np.concatenate(value, axis=0) if key != "X" else np.concatenate(value, axis=0).astype(np.float32)
        for key, value in arrays.items()
    }
    return output, records


def write_outputs(output_file: Path, arrays: dict[str, np.ndarray], records: list[dict], args: argparse.Namespace) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **arrays)
    print(f"Saved: {output_file}")
    print(f"X shape: {arrays['X'].shape}")


def main() -> None:
    args = parse_args()
    configure_data_cache(args.data_root)
    from moabb.datasets import BNCI2014_001

    subjects = resolve_subjects(BNCI2014_001(), args.subjects)
    arrays, records = build_dataset(subjects)
    write_outputs(args.output_file, arrays, records, args)


if __name__ == "__main__":
    main()
