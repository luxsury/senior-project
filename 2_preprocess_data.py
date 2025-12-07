import os
import json
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "processed_keypoints")
OUT_DIR = os.path.join(BASE_DIR, "data", "dataset")
os.makedirs(OUT_DIR, exist_ok=True)

# Augment settings
ENABLE_AUGMENT = True
AUG_PER_SAMPLE = 2  # 每個原始樣本產生幾個 augmented samples

def time_warp(seq, scale_range=(0.8, 1.2)):
    # uniform resample to simulate speed up / slow down
    n, d = seq.shape
    if n == 0: return seq.copy()
    scale = np.random.uniform(*scale_range)
    new_n = max(1, int(np.round(n * scale)))
    idx = np.linspace(0, n - 1, new_n).astype(int)
    out = seq[idx]
    return out

def add_noise(seq, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

def time_reverse(seq):
    return seq[::-1]

def random_crop(seq, min_ratio=0.6):
    n, d = seq.shape
    if n == 0: return seq.copy()
    crop_len = max(1, int(n * np.random.uniform(min_ratio, 1.0)))
    start = np.random.randint(0, n - crop_len + 1)
    return seq[start:start+crop_len]

def augment_sequence(seq):
    funcs = [time_warp, add_noise, time_reverse, random_crop]
    s = seq.copy()
    # apply 1-2 random ops
    ops = np.random.choice(funcs, size=np.random.randint(1,3), replace=False)
    for op in ops:
        s = op(s)
    return s

if __name__ == "__main__":
    classes = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    label_to_idx = {label: i for i, label in enumerate(classes)}
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(label_to_idx, f, indent=4)
    X = []
    y = []

    print("Detected Classes:", classes)

    for label in classes:
        class_path = os.path.join(RAW_DIR, label)
        for f in sorted(os.listdir(class_path)):
            if not f.endswith(".npy"): continue
            arr = np.load(os.path.join(class_path, f))
            # arr shape: (n_frames, feat_dim)
            X.append(arr.astype(np.float32))
            y.append(label_to_idx[label])
            print(f"Loaded: {os.path.join(class_path, f)}")

            if ENABLE_AUGMENT:
                for k in range(AUG_PER_SAMPLE):
                    aug = augment_sequence(arr)
                    X.append(aug.astype(np.float32))
                    y.append(label_to_idx[label])

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=np.int32)

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    print("Saved dataset to", OUT_DIR)
    print("Total samples:", len(X))
