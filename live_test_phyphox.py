import time
import requests
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# =========================
# SETTINGS (must match training)
# =========================
SAMPLE_RATE = 62.5
WINDOW_MS = 1000
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_MS / 1000)  # ~62 samples

# how often we classify (in seconds)
PREDICT_EVERY = 0.2  # 200ms

# smoothing window (majority vote)
SMOOTH_N = 7

# confidence threshold
CONF_THRESHOLD = 0.70

# =========================
# CHANGE THIS:
# phyphox Remote Access URL
# Example: http://192.168.43.127:8080
# =========================
PHONE_URL = "http://10.100.228.241"


# =========================
# Load trained model + scaler + labels
# =========================
model = tf.keras.models.load_model("output/gesture_model.h5")
labels = np.load("output/labels.npy", allow_pickle=True)

scaler_mean = np.load("output/scaler_mean.npy")
scaler_scale = np.load("output/scaler_scale.npy")

print(" Model loaded")
print(" LABEL ORDER:", labels)


# =========================
# Feature extraction (must match train_gesture_model.py)
# =========================
def extract_features(window):
    """
    window shape: (WINDOW_SIZE, 3)
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


def scale_features(x):
    """
    Apply same StandardScaler used during training
    """
    return (x - scaler_mean) / scaler_scale


# =========================
# phyphox data fetch
# =========================
def get_latest_accel():
    """
    Reads latest acceleration values from phyphox.
    Returns: (accX, accY, accZ)
    """
    url = PHONE_URL + "/get?accX&accY&accZ"
    r = requests.get(url, timeout=3)
    data = r.json()

    ax = data["buffer"]["accX"]["buffer"][-1]
    ay = data["buffer"]["accY"]["buffer"][-1]
    az = data["buffer"]["accZ"]["buffer"][-1]

    return float(ax), float(ay), float(az)


def pretty_probs(probs):
    """
    Print probabilities nicely: circle:0.92 shake:0.03 updown:0.05
    """
    parts = []
    for i, p in enumerate(probs):
        parts.append(f"{labels[i]}:{p:.2f}")
    return "  ".join(parts)


# =========================
# LIVE TEST MAIN
# =========================
def main():
    print("\n============================")
    print(" LIVE TEST STARTED")
    print("Perform gestures on phone:")
    print("- circle")
    print("- shake")
    print("- updown")
    print("============================\n")

    buffer = deque(maxlen=WINDOW_SIZE)           # stores [ax, ay, az]
    recent_preds = deque(maxlen=SMOOTH_N)        # store last predictions

    last_predict_time = 0

    while True:
        try:
            # get latest accel sample
            ax, ay, az = get_latest_accel()
            buffer.append([ax, ay, az])

            # wait until we have full window
            if len(buffer) < WINDOW_SIZE:
                print(f"Collecting samples... {len(buffer)}/{WINDOW_SIZE}", end="\r")
                time.sleep(0.02)
                continue

            # predict every 200ms
            if time.time() - last_predict_time < PREDICT_EVERY:
                continue

            last_predict_time = time.time()

            window = np.array(buffer, dtype=np.float32)

            # extract features and scale
            feats = extract_features(window)
            feats = scale_features(feats)

            # predict
            probs = model.predict(feats.reshape(1, -1), verbose=0)[0]
            best_idx = int(np.argmax(probs))
            best_label = str(labels[best_idx])
            best_conf = float(probs[best_idx])

            # confidence threshold
            if best_conf < CONF_THRESHOLD:
                recent_preds.append("unknown")
            else:
                recent_preds.append(best_label)

            # smoothing: majority vote
            vote = Counter(recent_preds).most_common(1)[0][0]

            print(
                f"Raw: {best_label:7s} ({best_conf:.2f})  "
                f"Smoothed: {vote:7s}   |   {pretty_probs(probs)}"
            )

        except KeyboardInterrupt:
            print("\n\n Stopped live testing.")
            break

        except Exception as e:
            print("\n Error:", e)
            time.sleep(1)


if __name__ == "__main__":
    main()
