# 🐘 Wildlife Detection and Alert System using Deep Learning

## 📌 Overview

This project presents an intelligent **Wildlife Detection and Alert System** designed to detect animals in real-time using advanced Deep Learning models. The system integrates multiple architectures including **YOLO, VGG19, Faster R-CNN, and a Hybrid Model** to improve detection accuracy and reliability.

The solution aims to reduce **human-wildlife conflicts** by generating alerts when animals are detected near sensitive zones such as highways, railways, and forest boundaries.

---

## 🚀 Key Features

* 🔍 Real-time wildlife detection from images/videos
* 🧠 Multiple Deep Learning models:

  * YOLO (You Only Look Once)
  * VGG19 (Classification)
  * Faster R-CNN (Object Detection)
  * Hybrid Model (Combined approach)
* 📊 Improved accuracy using model comparison & fusion
* 🌐 Interactive Web Application using Streamlit
* 📩 Instant alert system via Telegram
* 🌙 Works for both day and night scenarios (trained on diverse dataset)

---

## 🧠 Models Used

### 1. YOLO

* Fast, real-time object detection
* Suitable for live monitoring systems

### 2. VGG19

* Deep CNN used for image classification
* Helps in identifying animal categories

### 3. Faster R-CNN

* High accuracy object detection model
* Used for precise bounding box detection

### 4. Hybrid Model

* Combines strengths of multiple models
* Improves detection performance and robustness

---

## 📂 Project Structure

```
wildlife-detection-project/
│
│── yolo.ipynb
│── vgg19.ipynb
│── faster_rcnn.ipynb
├── hybrid_model.ipynb
│
├── webapp/
│   ├── app.py
│   ├── requirements.txt
│
├── dataset/
│   └── README.md  (dataset links only)
│
├── assets/ (optional screenshots)
├── README.md
└── .gitignore
```

---

## 📊 Dataset

Due to large size, datasets are not included in this repository.

📎 Dataset Links:

* Classification Dataset : https://universe.roboflow.com/dl-bdqrj/my-first-project-0f0ms
* Object Detection Dataset: https://universe.roboflow.com/bhaswanths-workspace-krsji/my-first-project-ogpvt

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/wildlife-detection-alert-system.git
cd wildlife-detection-alert-system
```

Install dependencies:

```bash
pip install -r webapp/requirements.txt
```

---

## ▶️ Run the Web Application

```bash
cd webapp
streamlit run app.py
```

---

## 📸 Results

* Accurate detection of animals in real-time
* Bounding boxes with confidence scores
* Alerts triggered for detected wildlife

(Add screenshots in `/assets` folder and link here)

---

## 📩 Alert System

* Integrated with Telegram Bot API
* Sends instant alerts when wildlife is detected
* Useful for forest monitoring and safety systems

* Telegram Bot Link: https://t.me/Wildlife_animals_alert_systembot
---

## 📈 Applications

* Forest border monitoring
* Railway track safety
* Highway accident prevention
* Smart surveillance systems
* Wildlife conservation research

---

## 🔮 Future Improvements

* Integration with thermal imaging for low-light detection
* Deployment on edge devices (NVIDIA Jetson Nano, Google Coral)
* Expansion to more animal classes
* Offline detection for remote areas
* Model optimization for faster inference

---

## 🛠️ Technologies Used

* Python
* OpenCV
* PyTorch / TensorFlow / Keras
* Streamlit
* NumPy, Pandas
* Deep Learning Models (YOLO, VGG19, Faster R-CNN)

## ⭐ Acknowledgements

* Open-source Deep Learning communities
* Research papers and datasets used for training

---

## 📜 License

This project is for academic and research purposes.

---
