# AR Navigation System for the Visually Impaired

## Overview
The **AR Navigation System for the Visually Impaired** is an AI-powered guidance solution that integrates **Augmented Reality (AR), Artificial Intelligence (AI), and sensor fusion** to provide real-time navigation assistance. The system offers **audio and haptic feedback**, enabling visually impaired users to navigate safely and efficiently in both indoor and outdoor environments.

## Features
- **Real-time Video Processing**: Captures and stabilizes camera feed at **15-30 FPS**.
- **Voice Command Support**: Uses **Automatic Speech Recognition (ASR)** models like Whisper and DeepSpeech.
- **Sensor Fusion**: Integrates **GPS, accelerometer, and gyroscope** for accurate positioning.
- **Object Detection**: Identifies objects using **YOLOv8, EfficientDet**.
- **Traffic Light & Sign Recognition**: Uses **EasyOCR** and classification models.
- **Depth Estimation**: Estimates obstacle distance with **MiDaS**.
- **Path Segmentation**: Detects walkable paths.
- **Path Planning & Obstacle Avoidance**: Uses **OpenStreetMap API** and **A* algorithm**.
- **Dynamic Path Recalculation**: Updates every **5-10 seconds** for real-time adaptability.
- **Audio & Haptic Feedback**: Provides spatial cues and vibration patterns.
- **Mobile Optimization**: Includes **quantization, pruning, TensorFlow Lite** for efficient edge processing.

## System Architecture
The system consists of four core modules:
1. **Input Processing**
   - Video stabilization
   - ASR-based voice commands
   - Sensor fusion with Kalman filtering
2. **Perception**
   - Object detection & recognition
   - Depth estimation
   - Path segmentation
3. **Navigation**
   - A* path planning
   - Real-time geospatial processing
   - Dynamic path recalculation
   - Safety buffers for obstacle avoidance
4. **User Interface**
   - Text-to-Speech (TTS) via ElevenLabs
   - Haptic feedback with adaptive preferences

## Implementation Details
- **Model Training**: Pre-trained on general datasets, fine-tuned for visually impaired navigation.
- **Real-time Processing**: Ensures **latency <150ms** for seamless guidance.
- **Evaluation Metrics**:
  - Object Detection Accuracy: **>90%**
  - Navigation Accuracy: **<3m deviation**
  - Response Time: **<200ms** for obstacle alerts
  - Power Efficiency: **3-4% per minute** battery consumption

## Tech Stack
- **Programming Languages**: Python, TensorFlow, PyTorch
- **Computer Vision**: OpenCV, YOLOv8, MiDaS, EfficientDet
- **Navigation & Mapping**: OpenStreetMap API, A*
- **Audio Processing**: Whisper, DeepSpeech, ElevenLabs
- **Hardware**: Edge computing devices, mobile platforms

## Future Enhancements
- **Integration with Smart Glasses**
- **Enhanced Indoor Navigation** with LiDAR
- **Multi-language Voice Support**
- **Community-based obstacle reporting**

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/AR-Navigation-System.git
   cd AR-Navigation-System
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Contributors
- **[Acchu]** - AI/ML Engineer

## License
This project is licensed under the **MIT License**.

## Contact
For any queries or contributions, reach out via **[harishreddy.workmail@gmail.com]**.

