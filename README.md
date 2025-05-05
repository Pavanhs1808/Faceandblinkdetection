# Blink Detection and Emotion Estimation with Face Tracking

This project uses OpenCV and MediaPipe to detect eye blinks in real-time from a webcam feed and estimates the user's emotion based on blink frequency. It also tracks the face using a bounding box for better visual feedback.

## 🔍 Features

* ✅ Real-time **blink detection** using Eye Aspect Ratio (EAR)
* ✅ **Face tracking** with a blue bounding box using OpenCV Haar cascades
* ✅ **Emotion estimation** based on blink rate over 30-second intervals
* ✅ Uses **MediaPipe FaceMesh** for high-precision facial landmarks

## 😊 Emotion Interpretation Based on Blink Rate

| Blink Rate (blinks/sec) | Estimated Emotion |
| ----------------------- | ----------------- |
| `< 0.09`                | Relaxed           |
| `0.09 - 0.2`            | Neutral           |
| `> 0.2`                 | Stressed          |

## 📦 Requirements

* Python 3.10+
* Webcam
* pip (Python package manager)

## 🛠 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/blink-emotion-detector.git
   cd blink-emotion-detector
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install opencv-python mediapipe
   ```

> **Note:** MediaPipe may not support Python 3.11+ officially. If you run into issues, try using Python 3.10.

## 🚀 How to Run

1. Plug in your webcam.
2. Run the script:

   ```bash
   python blink_detector.py
   ```
3. Press `q` to quit the application.

## 📂 File Structure

```
blink-emotion-detector/
├── blink_detector.py       # Main script
├── README.md               # Project README
```

## 🧠 How It Works

* Facial landmarks are detected using **MediaPipe FaceMesh**
* EAR (Eye Aspect Ratio) is calculated to determine if the eyes are closed
* If eyes are closed for multiple frames, a blink is registered
* Every 30 seconds, the blink frequency is analyzed and categorized into one of three emotions

## 📸 Sample Output

* Blue box surrounds the detected face
* Live blink count shown on screen
* Console prints emotion every 30 seconds
