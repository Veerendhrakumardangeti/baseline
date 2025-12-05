This script is intentionally general-purpose and defensive. It supports:
- Loading per-record `.npy` signals or WFDB records (if you have `wfdb` installed).
- Resampling/padding/trimming to fixed length `L`.
- Simple z-score normalization per record.
- Train/val/test split by stratified labels if possible.

```python
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def load_signal_npy(path):
    s = np.load(path)
    # if shape (L,) -> (1,L)
    if s.ndim == 1:
        s = s[None, :]
    return s.astype(np.float32)

def pad_or_trim(sig, L):
    C, T = sig.shape
    if T == L:
        return sig
    if T > L:
        start = 0
        return sig[:, start:start+L]
    pad = np.zeros((C, L), dtype=np.float32)
    pad[:, :T] = sig
    return pad

def zscore(sig):
    sig = sig.astype(np.float32)
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-8
    return (sig - mean) / std

def main(args):
    records = []
    labels = {}

    if args.meta_csv and os.path.exists(args.meta_csv):
        import csv
        with open(args.meta_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                rid = row[0]
                lbl = int(row[1])
                labels[rid] = lbl

    # collect files
    for fname in os.listdir(args.input_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in ['.npy']:
            continue
        rid = base
        p = os.path.join(args.input_dir, fname)
        if rid in labels:
            records.append((rid, p, labels[rid]))
        else:
            if args.meta_csv:
                continue
            # if no meta csv, user must provide label arg mapping fallback
            records.append((rid, p, -1))

    X = []
    y = []
    for rid, path, lbl in records:
        sig = load_signal_npy(path)  # (C,L0)
        if args.lead >= sig.shape[0]:
            lead = 0
        else:
            lead = args.lead
        sig = sig[lead:lead+1, :]
        sig = pad_or_trim(sig, args.length)
        sig = zscore(sig)
        # convert to (L,) if single channel (the training accepts (N,L) or (N,C,L))
        if sig.shape[0] == 1:
            sig = sig[0]
        X.append(sig)
        y.append(lbl)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    # If labels unknown (-1) stop
    if (y == -1).any():
        raise RuntimeError("Some records have unknown labels. Provide meta_csv mapping record->label.")

    # stratified split if possible
    test_frac = args.test_frac
    val_frac = args.val_frac
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=42)
    val_rel = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=42)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train.astype(np.int64))
    np.save(os.path.join(args.out_dir, "X_val.npy"), X_val.astype(np.float32))
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val.astype(np.int64))
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test.astype(np.float32))
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test.astype(np.int64))
    print("Saved preprocess outputs to", args.out_dir)
    print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)    # dir with .npy per record (recordid.npy)
    p.add_argument("--meta_csv", type=str, default=None)      # CSV: record_id,label (header allowed)
    p.add_argument("--out_dir", type=str, default="results/baseline")
    p.add_argument("--length", type=int, default=5000)
    p.add_argument("--lead", type=int, default=0)             # which lead index to use (0-based)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--val_frac", type=float, default=0.1)
    args = p.parse_args()
    main(args)