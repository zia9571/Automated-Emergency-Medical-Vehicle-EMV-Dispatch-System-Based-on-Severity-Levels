# Automated Emergency Medical Vehicle (EMV) Dispatch System Based on Severity Levels 🚑

This project aims to develop an **Automated Emergency Medical Vehicle Dispatch System** that detects accidents in real time, assesses their severity, and prioritizes emergency response based on severity levels. Using a hybrid deep learning approach, this system leverages **YOLOv8** and **Optical Flow** models to ensure accurate detection and classification of accidents.

---

## Features 🌟
- **Real-Time Accident Detection**: Detects accidents using CCTV footage.
- **Severity Classification**: Classifies accidents into predefined severity levels.
- **Multi-Class Object Detection**: Detects vehicles (cars, trucks, motorcycles) alongside accidents.
- **Automated Dispatch**: Prioritizes emergency medical response based on accident severity.

---

## Dataset 📊
- **Source**: Data collected from Kaggle, YouTube, and Roboflow. 
- **Classes**: 
  - Cars
  - Trucks
  - Motorcycles
  - Accidents
  - Severe Accidents
- **Annotation**: Annotated using Roboflow.
- **Size**: 
  - Total images: 4022
  - Training set: 2815 (75%)
  - Testing set: 604 (15%)
  - Validation set: 603 (15%)
- **Image Resolution**: 640x640 pixels.

---

## Hardware Requirements 🖥️
- **CPU**: Intel Core i5 (8th gen) or AMD Ryzen 5 (2nd gen).
- **RAM**: 8GB or more.
- **GPU**: NVIDIA GTX 1650 or similar with CUDA 12.1 and cuDNN support.
- **Storage**: 5GB of free disk space.

---

## Software Requirements 🛠️
- **Operating System**: Windows 10, macOS, or Ubuntu 18.04.
- **Programming Language**: Python 3.12.6 (64-bit).
- **Frameworks**:
  - Ultralytics 8.3.49
  - TensorFlow 2.18.0
  - PyTorch 2.6.0
  - Torchvision 0.20.0
- **Libraries**:
  - OpenCV 4.10.0
  - NumPy 2.0.2
  - SciPy 1.14.1
- **IDE**: Visual Studio Code with Python and Jupyter extensions.

---

## Model Training ⚙️
- **Epochs**: 200
- **Learning Rate**: 0.01
- **Training Duration**: ~240 minutes.
- **Checkpointing**: Periodic saving during training.
- **Final Model Format**: Saved as `.pt`.

---

## Results and Analysis 📈
The system successfully detects and classifies objects into the five predefined classes. Accurate bounding boxes are placed around objects of interest, and accident severity is assessed effectively.
Further results and thesis documentation is available at https://docs.google.com/document/d/1q3AR5EzI8KZX6hlY1C7yG-OgyW_oN_z71m2FuPin65Y/edit?usp=sharing

---

## How to Run 🚀
1. Clone the repository:
   ```bash
   git clone https://github.com/Automated-Emergency-Medical-Vehicle-EMV-Dispatch-System-Based-on-Severity-Levels.git
   
---
## Collaborators 👥

This project is made possible with contributions from:

- [Bikram Tripathi](https://github.com/Funza07)
- [Toni Deori](https://github.com/Toni-deori))]
- [Zahra Shaikh](https://github.com/zia9571)
