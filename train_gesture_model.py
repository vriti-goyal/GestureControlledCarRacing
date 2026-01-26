import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf



# SETTINGS

SAMPLE_RATE = 62.5
WINDOW_MS = 1000
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_MS / 1000)  # ~62 samples

STRIDE_MS = 200
STRIDE_SIZE = max(1, int(SAMPLE_RATE * STRIDE_MS / 1000))  # ~12 samples

DATA_DIR = "data"
OUTPUT_DIR = "output"


# Feature 
def extract_features(window):
    """
    window shape: (WINDOW_SIZE, 3) => [accX, accY, accZ]
    returns: 1D feature vector
    """
    feats = []
    for axis in range(3):
        a = window[:, axis]

        feats.append(np.mean(a))
        feats.append(np.std(a))
        feats.append(np.min(a))
        feats.append(np.max(a))
        feats.append(np.median(a))
        feats.append(np.sqrt(np.mean(a ** 2)))  # RMS
    return np.array(feats, dtype=np.float32)


def window_signal(arr, window_size, stride_size):
    """
    arr shape: (N, 3)
    returns list of windows, each shape: (window_size, 3)
    """
    windows = []
    for start in range(0, len(arr) - window_size + 1, stride_size):
        w = arr[start:start + window_size]
        windows.append(w)
    return windows


# ======================
# Helper: auto detect acceleration columns
# ======================
def find_col(df, keyword):
    """
    Return first column name that contains the keyword (case-insensitive)
    """
    for c in df.columns:
        if keyword.lower() in str(c).lower():
            return c
    return None


def load_dataset(data_dir):
    X = []
    y = []

    labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    labels.sort()

    total_files = 0
    used_files = 0

    for label in labels:
        label_dir = os.path.join(data_dir, label)

        for file in os.listdir(label_dir):
            if not file.endswith(".csv"):
                continue

            total_files += 1
            path = os.path.join(label_dir, file)

            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"Skipping unreadable file: {path}\nReason: {e}")
                continue

            # ---- AUTO DETECT phyphox acceleration columns ----
            accx_col = find_col(df, "Acceleration x")
            accy_col = find_col(df, "Acceleration y")
            accz_col = find_col(df, "Acceleration z")

            if not (accx_col and accy_col and accz_col):
                print(f"\n Skipping file (missing acceleration columns): {path}")
                print("Columns found:", df.columns.tolist())
                continue

            # Keep only acceleration columns
            df = df[[accx_col, accy_col, accz_col]].dropna()
            df.columns = ["accX", "accY", "accZ"]

            # Convert to numeric
            df["accX"] = pd.to_numeric(df["accX"], errors="coerce")
            df["accY"] = pd.to_numeric(df["accY"], errors="coerce")
            df["accZ"] = pd.to_numeric(df["accZ"], errors="coerce")
            df = df.dropna()

            arr = df.to_numpy(dtype=np.float32)

            # Must have enough rows
            if len(arr) < WINDOW_SIZE:
                print(f"⚠️ Skipping short file (<{WINDOW_SIZE} rows): {path}")
                continue

            used_files += 1

            # Create windows
            windows = window_signal(arr, WINDOW_SIZE, STRIDE_SIZE)

            # Extract features
            for w in windows:
                X.append(extract_features(w))
                y.append(label)

    if len(X) == 0:
        raise ValueError("❌ No training windows created. Check your dataset / CSV files.")

    X = np.vstack(X)
    y = np.array(y)

    print(f"\n✅ Files scanned: {total_files}")
    print(f"✅ Files used:    {used_files}")
    print(f"✅ Total windows: {len(X)}\n")

    return X, y


def main():
    print("Loading dataset...")
    X, y = load_dataset(DATA_DIR)
    print("X shape:", X.shape)  # (num_windows, num_features)
    print("y shape:", y.shape)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("Classes:", list(le.classes_))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build neural network (TinyML-friendly)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(len(le.classes_), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining model...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

    print("\nTesting model...")
    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(os.path.join(OUTPUT_DIR, "gesture_model.h5"))
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), le.classes_)
    np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "scaler_scale.npy"), scaler.scale_)

    print("\n✅ Saved files in output/:")
    print("- gesture_model.h5")
    print("- labels.npy")
    print("- scaler_mean.npy")
    print("- scaler_scale.npy")


if __name__ == "__main__":
    main()
