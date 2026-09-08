#!/usr/bin/env python3
"""Quick smoke test: download 1 subject, verify epochs extraction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ntab_bci.data import load_subject_epochs

print("Testing data download + preprocessing...")
print("Subject 1, runs [4, 8, 12] (imagined left/right fist)")
print()

epochs = load_subject_epochs(
    subject=1,
    runs=[4, 8, 12],
    data_dir="data/eegmmidb",
    tmin=0.5,
    tmax=3.5,
    l_freq=8.0,
    h_freq=30.0,
)

print(f"✓ Downloaded and processed successfully")
print(f"  Epochs: {len(epochs)}")
print(f"  Channels: {len(epochs.ch_names)}")
print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
print(f"  Epoch duration: {epochs.tmax - epochs.tmin:.2f}s")
print()
print(f"Event counts:")
for event_name, event_id in epochs.event_id.items():
    count = (epochs.events[:, 2] == event_id).sum()
    print(f"  {event_name}: {count}")
print()
print("Pipeline is working! Ready to run scripts/train.py")
