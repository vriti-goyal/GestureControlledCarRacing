# Gesture Recognition Based Mood Light (TinyML Project)

## Project Overview

This project implements a gesture recognition system using accelerometer sensor data and a lightweight machine learning model (TinyML). Phone gestures such as Circle, Shake, and Up–Down are detected in real time using accelerometer values (X, Y, Z axes). Based on the detected gesture, the system can be extended to control a Mood Light (for example: rainbow mode, party mode, brightness control).

The project demonstrates a complete TinyML workflow without using platforms like Edge Impulse:

- Sensor data collection
- Windowing and feature extraction
- Neural network training using Python
- Live inference using phone sensors (via phyphox)
- Real-time gesture classification

All inference is performed locally on the edge (laptop), with no cloud dependency.

---

## Objectives

- Learn how accelerometer data can be used for gesture recognition
- Build a lightweight neural network suitable for TinyML
- Perform real-time gesture classification
- Understand the full ML pipeline from raw sensor data to live prediction

---

## What is TinyML in this Project?

TinyML refers to deploying small machine learning models on edge devices.

In this project:

- A compact neural network is trained
- Only 18 statistical features are used per gesture window
- Predictions are performed locally (no cloud)
- The model can be converted to TensorFlow Lite for microcontroller deployment

Hence, this project follows TinyML principles:
- Lightweight model
- Sensor-based input
- Edge inference
- Real-time operation

---

## Folder Structure

GestureLightML/
│
├── data/
│ ├── circle/ # CSV gesture samples (circle)
│ ├── shake/ # CSV gesture samples (shake)
│ └── updown/ # CSV gesture samples (updown)
│
├── live_raw/ # Optional raw phyphox exports
│
├── output/
│ ├── gesture_model.h5 # Trained neural network model
│ ├── labels.npy # Label mapping
│ ├── scaler_mean.npy # Feature scaling mean
│ └── scaler_scale.npy # Feature scaling standard deviation
│
├── train_gesture_model.py # Model training script
├── live_test_phyphox.py # Live gesture testing using phone accelerometer
└── README.md


---

## File Descriptions

### train_gesture_model.py

This script performs model training:

- Reads CSV accelerometer files from the data folder
- Automatically detects accelerometer columns
- Segments signals into 1-second sliding windows
- Extracts statistical features (mean, std, min, max, median, RMS)
- Normalizes features using StandardScaler
- Trains a neural network using TensorFlow
- Evaluates accuracy using classification report and confusion matrix
- Saves the trained model and preprocessing files in the output folder

Main libraries used:
- pandas (CSV handling)
- numpy (numerical operations)
- scikit-learn (scaling, splitting, evaluation)
- tensorflow (neural network)

Run training:

bash
python train_gesture_model.py
live_test_phyphox.py

This script performs live gesture recognition:

Connects to the phone via phyphox Remote Access over Wi-Fi

Continuously reads accelerometer values (X, Y, Z)

Buffers 1 second of sensor data

Extracts the same features used during training

Applies saved normalization

Runs model inference in real time

Displays predicted gesture and confidence

Uses majority voting to stabilize predictions

Main libraries used:

requests (HTTP communication with phone)

tensorflow (model inference)

numpy (feature processing)

collections (buffering and smoothing)

Run live testing:

python live_test_phyphox.py

Data Collection

Gesture data is collected using the phyphox mobile application:

Open phyphox → Accelerometer

Perform a gesture for approximately 4–5 seconds

Export raw_data.csv

Place files into:

data/circle/

data/shake/

data/updown/

Balanced datasets improve classification accuracy.

Machine Learning Pipeline

Collect accelerometer samples

Segment into 1-second windows

Extract statistical features

Normalize features

Train neural network

Save model

Perform live inference

Predict gestures in real time

Model Architecture

Input: 18 features

Dense Layer: 20 neurons (ReLU)

Dense Layer: 10 neurons (ReLU)

Output Layer: 3 classes (Softmax)

This lightweight architecture is suitable for TinyML deployment.

Example Output
Raw: circle (0.86)  Smoothed: circle
circle:0.86  shake:0.07  updown:0.07

Possible Extensions

Add graphical Mood Light interface (rainbow / party modes)

Convert model to TensorFlow Lite

Deploy on Arduino Nano 33 BLE Sense or ESP32

Add gyroscope data for improved accuracy
