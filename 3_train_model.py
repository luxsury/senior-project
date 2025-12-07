import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
# -----------------------------------------------------------
# 路徑設定：可依環境調整
# -----------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")   # X.npy / y.npy / labels.json 放這裡
MODEL_DIR = os.path.join(BASE_DIR, "models")             # 模型輸出資料夾
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------
# 基本訓練參數（你主要會調整的區域）
# -----------------------------------------------------------
SEQ_LEN = 40            # 序列長度：可試 30 / 40 / 60 / 80
BATCH_SIZE = 8          # batch size：小資料集通常 8~16
EPOCHS = 60             # 訓練次數：常見 40~120
LR = 1e-3               # 初始 learning rate
RANDOM_STATE = 42        # 亂數種子，讓結果可重現
MODEL_TYPE = "conv_lstm"  # "lstm" 或 "conv_lstm"

# -----------------------------------------------------------
# 讀取原始資料
# -----------------------------------------------------------
def load_raw():
    X = np.load(os.path.join(DATA_DIR, "X.npy"), allow_pickle=True)
    y = np.load(os.path.join(DATA_DIR, "y.npy"), allow_pickle=True)
    with open(os.path.join(DATA_DIR, "labels.json"), "r") as f:
        label_to_idx = json.load(f)
    # 轉成 index -> label
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    return X, y, idx_to_label

# -----------------------------------------------------------
# 對每段序列進行等長取樣（或補零）
# -----------------------------------------------------------
def sample_or_pad(seq, seq_len):
    # 若為空序列
    if seq is None or getattr(seq, "size", 0) == 0:
        return np.zeros((seq_len, feat_dim), dtype=np.float32)
    n = seq.shape[0]

    # 長度 >= 目標，採樣
    if n >= seq_len:
        idx = np.linspace(0, n - 1, seq_len).astype(int)
        return seq[idx]
    # 長度不足 → 補零
    else:
        out = np.zeros((seq_len, seq.shape[1]), dtype=np.float32)
        out[:n] = seq
        return out

# -----------------------------------------------------------
# 將所有序列轉成固定長度矩陣 X
# -----------------------------------------------------------
def build_X_matrix(X_raw, seq_len):
    global feat_dim
    feat_dim = None

    # 找一個非空序列的 feature 維度
    for s in X_raw:
        if getattr(s, "size", 0) != 0:
            feat_dim = s.shape[1]
            break
    if feat_dim is None:
        raise ValueError("Can't infer feature dim.")

    X_out = np.zeros((len(X_raw), seq_len, feat_dim), dtype=np.float32)
    for i, s in enumerate(X_raw):
        X_out[i] = sample_or_pad(s, seq_len)
    return X_out

# -----------------------------------------------------------
# 計算 dataset 的均值 / 標準差（做 normalization）
# -----------------------------------------------------------
def compute_norm(X):
    all_frames = X.reshape(-1, X.shape[-1])
    mean = np.mean(all_frames, axis=0)
    std = np.std(all_frames, axis=0) + 1e-8
    return mean, std

# -----------------------------------------------------------
# 正規化
# -----------------------------------------------------------
def normalize(X, mean, std):
    return (X - mean.reshape((1,1,-1))) / std.reshape((1,1,-1))

# -----------------------------------------------------------
# 純 LSTM 模型
# -----------------------------------------------------------
def make_lstm(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

# -----------------------------------------------------------
# Conv + LSTM（通常較穩定）
# -----------------------------------------------------------
def make_conv_lstm(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)

    # Temporal Conv1D 卷積：增加局部時間特徵
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    # LSTM 抽取長期時間特徵
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

# -----------------------------------------------------------
# 主流程
# -----------------------------------------------------------
def main():
    X_raw, y, idx_to_label = load_raw()
    print("Loaded", len(X_raw), "samples")

    # 將序列處理成固定長度
    X = build_X_matrix(X_raw, SEQ_LEN)
    print("X shape:", X.shape, "y:", y.shape)

    # 資料切割 train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    # 計算 normalization 參數
    mean, std = compute_norm(X_train)
    np.savez(os.path.join(MODEL_DIR, "scaler.npz"), mean=mean, std=std)

    # 套用 normalization
    X_train = normalize(X_train, mean, std)
    X_val = normalize(X_val, mean, std)

    # 自動計算 class weight（若某些動作樣本少）
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): w for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

    # 選擇模型
    if MODEL_TYPE == "lstm":
        model = make_lstm(X_train.shape[1:], len(idx_to_label))
    else:
        model = make_conv_lstm(X_train.shape[1:], len(idx_to_label))

    model.summary()

    # callback：早停 + 降 LR + 儲存最好的模型
    ckpt = os.path.join(MODEL_DIR, "pose_sequence_classifier.h5")
    cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    cb_ckpt = callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)
    cb_rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    # tf.data pipeline
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(512, seed=RANDOM_STATE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 訓練模型
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[cb_early, cb_ckpt, cb_rl]
    )

    # 儲存 label map
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        json.dump({int(k): v for k, v in idx_to_label.items()}, f, indent=4)

    print("Saved model & scaler in", MODEL_DIR)


if __name__ == "__main__":
    main()
