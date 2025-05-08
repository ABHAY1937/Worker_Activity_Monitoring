# **Optimized Computer Vision System for Worker Activity Monitoring**

This project provides a real-time computer vision system to monitor worker activity using a video stream or RTSP input. The system is built with Python 3.8, powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), and has been tested on an RTX 4060 (8GB) GPU.

---

## 📦 Repository

👉 [GitHub Link](https://github.com/ABHAY1937/Worker_Activity_Monitoring.git)
---

## 📹 Overview

- Accepts input via RTSP stream or local video file
- Automatically runs on system startup using `systemd`
- Outputs annotated video for monitoring worker behavior
- Ideal for safety, compliance, and activity tracking scenarios

---

## 🧰 Requirements

Install Ultralytics directly:

```bash
pip install ultralytics

./setup.sh
