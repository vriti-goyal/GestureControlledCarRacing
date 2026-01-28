# GestureLightML - Gesture-Based Car Game

A machine learning project that trains gesture recognition models on smartphone accelerometer data and uses them to control a car in a Pygame-based game using real-time gesture detection.

## Project Overview

This project captures gesture data from a smartphone (via Phyphox), trains a neural network to recognize gestures, and uses the trained model to control a car in a game. The supported gestures are:
- **Circle**: Circular motion detection
- **Shake**: Rapid shaking motion
- **UpDown**: Up and down motion

## Project Structure

```
GestureLightML/
├── README.md                          # This file
├── car.py                             # Car class for game mechanics
├── gesture_car_game.py                # Main game that uses gesture recognition
├── train_gesture_model.py             # Script to train the gesture model
├── live_test_phyphox.py               # Real-time gesture testing script
├── assets/                            # Game assets
│   ├── car.png                        # Car sprite image
│   └── car_files/                     # Additional car-related assets
├── data/                              # Training data (accelerometer CSV files)
│   ├── circle/                        # Circle gesture samples (9 files)
│   │   ├── circle1.csv
│   │   ├── circle2.csv
│   │   └── ... (up to circle10)
│   ├── shake/                         # Shake gesture samples (9 files)
│   │   ├── shake1.csv
│   │   ├── shake2.csv
│   │   └── ... (up to shake10)
│   └── updown/                        # Up/Down gesture samples (9 files)
│       ├── updown1.csv
│       ├── updown2.csv
│       └── ... (up to updown10)
├── output/                            # Trained model and preprocessing files
│   ├── gesture_model.h5               # Trained Keras model
│   ├── gesture_model.tflite           # TensorFlow Lite model (mobile-ready)
│   ├── labels.npy                     # Gesture labels
│   ├── scaler_mean.npy                # Mean values for feature scaling
│   └── scaler_scale.npy               # Scale values for feature normalization
└── __pycache__/                       # Python cache (auto-generated)
```

## File Descriptions

### Core Scripts

#### `train_gesture_model.py`
Trains the gesture recognition neural network from CSV accelerometer data.
- Reads accelerometer data from `data/` subdirectories
- Extracts 18-dimensional features from 1000ms windows
- Normalizes features using StandardScaler
- Trains a deep neural network using TensorFlow/Keras
- Saves model, labels, and scaler parameters to `output/`
- Generates confusion matrix and classification report

**Key Parameters:**
- `SAMPLE_RATE`: 62.5 Hz (from Phyphox)
- `WINDOW_SIZE`: ~62 samples (1000ms window)
- `STRIDE_SIZE`: ~12 samples (200ms overlap stride)

#### `gesture_car_game.py`
Main game that communicates with a smartphone via HTTP and uses gesture predictions to control a car.
- Connects to smartphone running Phyphox Remote Access
- Streams accelerometer data in real-time
- Applies trained model for gesture prediction
- Controls car movement (left/right lanes) based on gestures
- Renders game using Pygame

**Configuration:**
```python
PHONE_URL = "http://10.100.228.241"   # Change to your phone's IP
SAMPLE_RATE = 62.5
WINDOW_SIZE = ~62
CONF_THRESHOLD = 0.7                  # Minimum confidence for gesture recognition
SMOOTH_N = 5                          # Smoothing window size
```

#### `live_test_phyphox.py`
Testing script for real-time gesture recognition without the game.
- Connects to smartphone Phyphox instance
- Streams accelerometer data
- Applies trained model predictions
- Displays recognized gestures and confidence scores
- Includes smoothing for stable predictions

**Configuration:**
```python
PHONE_URL = "http://10.100.228.241"   # Change to your phone's IP
PREDICT_EVERY = 0.2                   # Prediction frequency (200ms)
SMOOTH_N = 7                          # Smoothing window for majority voting
CONF_THRESHOLD = 0.70                 # Confidence threshold
```

### Game Components

#### `car.py`
Defines the Car class for game mechanics.
- Manages car position across 3 lanes
- Handles left/right movement with smooth animation
- Properties:
  - `lane`: Current lane (0, 1, or 2)
  - `lane_x`: X-coordinates for each lane
  - `speed`: Smooth movement speed
  - `y`: Fixed vertical position

### Data Files

#### `data/` Directory
Contains raw accelerometer data in CSV format collected using Phyphox.
- Each CSV file contains 3 columns: `accX`, `accY`, `accZ` (acceleration values)
- 3 gesture categories with 10 samples each (30 total training samples)
- Data is used to train the gesture recognition model

#### `output/` Directory
Pre-trained models and preprocessing artifacts:
- **gesture_model.h5**: Keras neural network model
- **gesture_model.tflite**: TensorFlow Lite format for mobile deployment
- **labels.npy**: Array of gesture class names
- **scaler_mean.npy**: Mean values for feature normalization
- **scaler_scale.npy**: Standard deviation values for feature scaling

### Assets

#### `assets/` Directory
- **car.png**: Sprite image for the car in the game
- **car_files/**: Additional car-related assets

## Installation

### Prerequisites
- Python 3.7+
- Smartphone with Phyphox app installed (for real-time data)

### Dependencies
```bash
pip install pygame requests numpy tensorflow scikit-learn pandas
```

### Setup Steps

1. **Clone/Download the project**
   ```bash
   cd GestureLightML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # If available, or use pip install commands above
   ```

3. **Configure phone connection**
   - Install [Phyphox](https://phyphox.org/) on your smartphone
   - Enable Remote Access in Phyphox settings
   - Note your phone's IP address
   - Update `PHONE_URL` in `gesture_car_game.py` and `live_test_phyphox.py`

## Usage

### Training a New Model

If you have new accelerometer data, retrain the model:

```bash
python train_gesture_model.py
```

This will:
- Load all CSV files from `data/` subdirectories
- Train a neural network
- Save outputs to `output/`
- Display training metrics and confusion matrix

### Playing the Gesture Car Game

Make sure your phone is connected and Phyphox is running:

```bash
python gesture_car_game.py
```

**How to play:**
- Perform gestures with your phone to control the car
- **Circle gesture**: Move car to a specific lane
- **Shake gesture**: Quick lateral movement
- **UpDown gesture**: Alternative movement control
- Avoid obstacles while the car is moving

### Testing Gestures in Real-Time

To test gesture recognition without playing the game:

```bash
python live_test_phyphox.py
```

This will:
- Display recognized gestures in real-time
- Show confidence scores for each prediction
- Help verify model accuracy before gaming

## Feature Engineering

The model extracts 18 features from each 1-second accelerometer window (6 features × 3 axes):
- **Mean**: Average acceleration
- **Standard Deviation**: Variability
- **Minimum**: Lowest value
- **Maximum**: Highest value
- **Median**: Middle value
- **RMS**: Root Mean Square (magnitude)

These features are calculated for X, Y, and Z acceleration axes.

## Model Architecture

- **Input**: 18-dimensional feature vector
- **Layers**: Dense neural network with multiple hidden layers
- **Output**: Softmax classification (3 gesture classes)
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy

## Configuration Parameters

All key parameters are defined at the top of each script and should match between training and inference:

```python
SAMPLE_RATE = 62.5          # Hz (Phyphox sampling rate)
WINDOW_MS = 1000            # Milliseconds (feature extraction window)
STRIDE_MS = 200             # Milliseconds (window stride for training)
CONF_THRESHOLD = 0.7        # Minimum confidence for predictions
SMOOTH_N = 5-7              # Smoothing window size (majority voting)
```

## Troubleshooting

### Connection Issues
- Ensure phone and computer are on the same network
- Verify the `PHONE_URL` is correct (check Phyphox app)
- Check phone's IP address hasn't changed

### Model Accuracy
- Collect more training samples per gesture
- Ensure consistent gesture performance
- Check feature distributions in training data

### Performance Issues
- Reduce `SMOOTH_N` for faster response
- Increase `PREDICT_EVERY` for fewer predictions
- Check network latency

## Future Improvements

- Add more gesture types (swipe, tilt, etc.)
- Implement data augmentation for better generalization
- Add game difficulty levels
- Include obstacle avoidance mechanics
- Deploy model directly to smartphone

## Requirements

Create a `requirements.txt` file with:
```
pygame>=2.1.0
requests>=2.28.0
numpy>=1.21.0
tensorflow>=2.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

## License

This project is for educational purposes.

## Author

Created as a gesture recognition machine learning project.

---

**Last Updated**: January 2026
