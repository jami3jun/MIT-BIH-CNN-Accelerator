"""
loader.py
Loads raw MIT-BIH records using wfdb and returns signals + annotations.
"""

import wfdb
import numpy as np
from pathlib import Path

# All 48 MIT-BIH record IDs
ALL_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

# AAMI standard 5-class mapping
# N=Normal, S=Supraventricular, V=Ventricular, F=Fusion, Q=Unknown/Paced
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',   # Normal
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',               # Supraventricular
    'V': 'V', 'E': 'V',                                     # Ventricular
    'F': 'F',                                               # Fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q',                          # Unknown/Paced
}

LABEL_TO_INT = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}


def load_record(record_id: str, data_dir: str | Path) -> tuple:
    """
    Load a single MIT-BIH record.

    Returns:
        signal: np.ndarray of shape (num_samples, 2) — both leads
        ann_samples: np.ndarray of beat sample indices
        ann_symbols: list of beat annotation symbols
    """
    data_dir = Path(data_dir)
    record_path = str(data_dir / record_id)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal  # shape: (num_samples, num_leads)
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    return signal, ann_samples, ann_symbols


def get_beat_label(symbol: str) -> int | None:
    """Map a raw annotation symbol to an integer AAMI class. Returns None to skip."""
    aami = AAMI_MAP.get(symbol, None)
    if aami is None:
        return None
    return LABEL_TO_INT[aami]


def inspect_record(record_id: str, data_dir: str | Path):
    """Quick sanity check — print basic info about a record."""
    signal, ann_samples, ann_symbols = load_record(record_id, data_dir)

    print(f"Record {record_id}")
    print(f"  Signal shape : {signal.shape}")   # e.g. (650000, 2)
    print(f"  Sample rate  : 360 Hz")
    print(f"  Duration     : {signal.shape[0] / 360 / 60:.1f} minutes")
    print(f"  Total beats  : {len(ann_samples)}")

    from collections import Counter
    symbol_counts = Counter(ann_symbols)
    print(f"  Beat symbols : {dict(symbol_counts)}")