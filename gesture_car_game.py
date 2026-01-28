import pygame
import requests
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from car import Car

PHONE_URL = "http://10.100.228.241"   # CHANGE TO YOUR PHONE IP

SAMPLE_RATE = 62.5
WINDOW_SIZE = int(SAMPLE_RATE * 1000 / 1000)
CONF_THRESHOLD = 0.7
SMOOTH_N = 5

model = tf.keras.models.load_model("output/gesture_model.h5")
labels = np.load("output/labels.npy", allow_pickle=True)
scaler_mean = np.load("output/scaler_mean.npy")
scaler_scale = np.load("output/scaler_scale.npy")

def extract_features(window):
    feats = []
    for axis in range(3):
        a = window[:, axis]
        feats += [
            np.mean(a),
            np.std(a),
            np.min(a),
            np.max(a),
            np.median(a),
            np.sqrt(np.mean(a**2))
        ]
    return np.array(feats, dtype=np.float32)

def scale(x):
    return (x - scaler_mean) / scaler_scale

def get_latest_accel():
    url = PHONE_URL + "/get?accX&accY&accZ"
    r = requests.get(url, timeout=2)
    d = r.json()
    return (
        d["buffer"]["accX"]["buffer"][-1],
        d["buffer"]["accY"]["buffer"][-1],
        d["buffer"]["accZ"]["buffer"][-1]
    )

class GameEngine:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400,600))
        pygame.display.set_caption("Gesture Controlled Car")
        self.clock = pygame.time.Clock()

        self.car = Car()
        self.road_y = 0

        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.recent = deque(maxlen=SMOOTH_N)

        self.last_action = "none"

        self.car_img = pygame.image.load("assets/car.png")
        self.car_img = pygame.transform.scale(self.car_img,(60,100))

    def draw_road(self):
        self.screen.fill((40,40,40))
        for y in range(-80,600,40):
            pygame.draw.rect(self.screen,(255,255,255),(195,y+self.road_y,10,20))
        self.road_y += 6
        if self.road_y > 40:
            self.road_y = 0

    def get_gesture(self):
        try:
            ax,ay,az = get_latest_accel()
            self.buffer.append([ax,ay,az])

            if len(self.buffer) < WINDOW_SIZE:
                return "none"

            window = np.array(self.buffer)
            feats = extract_features(window)
            feats = scale(feats)

            probs = model.predict(feats.reshape(1,-1),verbose=0)[0]
            idx = np.argmax(probs)

            if probs[idx] < CONF_THRESHOLD:
                return "none"

            self.recent.append(labels[idx])
            return Counter(self.recent).most_common(1)[0][0]
        except:
            return "none"

    def run(self):

        running=True
        print("🎮 Game Started")
        print("UpDown = LEFT | Shake = RIGHT | Circle = STABLE")

        while running:

            for e in pygame.event.get():
                if e.type==pygame.QUIT:
                    running=False

            gesture = self.get_gesture()

            if gesture != self.last_action:

                # UpDown -> LEFT
                if gesture == "updown":
                    self.car.move_left()
                    print("LEFT")

                # Shake -> RIGHT
                elif gesture == "shake":
                    self.car.move_right()
                    print("RIGHT")

                # Circle -> STABLE (do nothing)
                elif gesture == "circle":
                    print("STABLE")

                self.last_action = gesture

            self.draw_road()
            x,y = self.car.get_position()
            self.screen.blit(self.car_img,(x,y))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

if __name__=="__main__":
    GameEngine().run()

