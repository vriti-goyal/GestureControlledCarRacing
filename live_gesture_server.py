import time
import requests
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from flask import Flask, jsonify
import threading

WINDOW_SIZE = 62
PREDICT_EVERY = 0.2
SMOOTH_N = 7
CONF_THRESHOLD = 0.7

PHONE_URL = "http://192.168.1.4"

model = tf.keras.models.load_model("output/gesture_model.h5")
labels = np.load("output/labels.npy", allow_pickle=True)

scaler_mean = np.load("output/scaler_mean.npy")
scaler_scale = np.load("output/scaler_scale.npy")

app = Flask(__name__)
current_gesture = "none"

@app.route("/gesture")
def gesture():
    return jsonify({"gesture": current_gesture})

def extract_features(w):
    f=[]
    for i in range(3):
        a=w[:,i]
        f += [np.mean(a),np.std(a),np.min(a),np.max(a),np.median(a),np.sqrt(np.mean(a*a))]
    return np.array(f)

def scale(x):
    return (x-scaler_mean)/scaler_scale

def get_acc():
    r=requests.get(PHONE_URL+"/get?accX&accY&accZ")
    d=r.json()
    return float(d["buffer"]["accX"]["buffer"][-1]),float(d["buffer"]["accY"]["buffer"][-1]),float(d["buffer"]["accZ"]["buffer"][-1])

def ml_loop():
    global current_gesture
    buf=deque(maxlen=WINDOW_SIZE)
    votes=deque(maxlen=SMOOTH_N)
    last=0

    while True:
        try:
            ax,ay,az=get_acc()
            buf.append([ax,ay,az])

            if len(buf)<WINDOW_SIZE: continue
            if time.time()-last<PREDICT_EVERY: continue

            last=time.time()

            w=np.array(buf)
            f=scale(extract_features(w))

            p=model.predict(f.reshape(1,-1),verbose=0)[0]
            idx=np.argmax(p)

            if p[idx]<CONF_THRESHOLD:
                votes.append("none")
            else:
                votes.append(labels[idx])

            current_gesture=Counter(votes).most_common(1)[0][0]
            print("Gesture:",current_gesture)

        except:
            time.sleep(0.3)

threading.Thread(target=ml_loop,daemon=True).start()

print("SERVER STARTED → http://localhost:5000/gesture")
app.run(host="0.0.0.0",port=5000)
