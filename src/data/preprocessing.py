"""
preprocessing.py
Takes raw signals and annotations from loader.py and produces
segmented, filtered, normalized beat windows ready for training.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from collections import Counter
from pathlib import Path

from src.data.loader import load_record, get_beat_label, ALL_RECORDS

# ── Constants ────────────────────────────────────────────────────────────────
FS = 360                # Sampling frequency (Hz)
WINDOW_SAMPLES = 300    # Total samples per beat window
BEFORE_PEAK = 150       # Samples to take before R-peak
AFTER_PEAK = WINDOW_SAMPLES - BEFORE_PEAK   # Samples after R-peak


# ── Step 1: Bandpass filter ───────────────────────────────────────────────────
def bandpass_filter(signal: np.ndarray, lowcut=0.5, highcut=40.0, fs=FS) -> np.ndarray:
    """
    Remove baseline wander (below 0.5 Hz) and high-frequency noise (above 40 Hz).
    signal shape: (num_samples, 2)
    """
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        filtered[:, lead] = filtfilt(b, a, signal[:, lead])
    return filtered


# ── Step 2: Segment into individual beat windows ─────────────────────────────
def segment_beats(signal: np.ndarray, ann_samples: np.ndarray,
                  ann_symbols: list) -> tuple:
    """
    Cut a fixed window around each annotated R-peak.
    Skips beats too close to the start/end of the recording.

    Returns:
        beats:  np.ndarray of shape (num_valid_beats, 2, WINDOW_SAMPLES)
        labels: np.ndarray of shape (num_valid_beats,) with integer AAMI labels
    """
    beats = []
    labels = []
    num_samples = signal.shape[0]

    for sample, symbol in zip(ann_samples, ann_symbols):
        # Get integer label — returns None for non-beat annotations (noise markers etc.)
        label = get_beat_label(symbol)
        if label is None:
            continue

        # Define window boundaries
        start = sample - BEFORE_PEAK
        end = sample + AFTER_PEAK

        # Skip beats too close to edges
        if start < 0 or end > num_samples:
            continue

        window = signal[start:end, :]       # shape: (WINDOW_SAMPLES, 2)
        window = window.T                   # shape: (2, WINDOW_SAMPLES) — channels first for PyTorch

        beats.append(window)
        labels.append(label)

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)


# ── Step 3: Normalize each beat window ───────────────────────────────────────
def normalize_beats(beats: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each beat window independently per lead.
    beats shape: (num_beats, 2, WINDOW_SAMPLES)
    """
    normalized = np.zeros_like(beats)
    for i in range(len(beats)):
        for lead in range(beats.shape[1]):
            channel = beats[i, lead, :]
            mean = channel.mean()
            std = channel.std()
            if std < 1e-6:          # Flat line guard
                std = 1.0
            normalized[i, lead, :] = (channel - mean) / std
    return normalized


# ── Full pipeline for one record ─────────────────────────────────────────────
def process_record(record_id: str, data_dir: str | Path) -> tuple:
    """
    Run the full preprocessing pipeline on a single record.

    Returns:
        beats:  np.ndarray (num_beats, 2, 300)
        labels: np.ndarray (num_beats,)
    """
    signal, ann_samples, ann_symbols = load_record(record_id, data_dir)
    signal = bandpass_filter(signal)
    beats, labels = segment_beats(signal, ann_samples, ann_symbols)
    beats = normalize_beats(beats)
    return beats, labels


# ── Process all records and save to disk ─────────────────────────────────────
def build_dataset(data_dir: str | Path, output_dir: str | Path,
                  records: list = ALL_RECORDS):
    """
    Process all records and save combined numpy arrays to output_dir.
    This only needs to be run once.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_beats = []
    all_labels = []
    all_record_ids = []   # Track which record each beat came from (needed for patient-wise splitting)

    for record_id in records:
        print(f"Processing record {record_id}...", end=" ")
        try:
            beats, labels = process_record(record_id, data_dir)
            all_beats.append(beats)
            all_labels.append(labels)
            all_record_ids.extend([record_id] * len(labels))
            print(f"{len(labels)} beats")
        except Exception as e:
            print(f"FAILED — {e}")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    record_ids = np.array(all_record_ids)

    # Print dataset summary
    print(f"\nDataset shape : {X.shape}")
    print(f"Labels shape  : {y.shape}")
    label_names = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
    counts = Counter(y.tolist())
    print("Class distribution:")
    for k, name in label_names.items():
        print(f"  {name}: {counts.get(k, 0)}")

    # Save
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "record_ids.npy", record_ids)
    print(f"\nSaved to {output_dir}")