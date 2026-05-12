# 🚗 Real-Time Lane & Edge Detection + YOLOv8 Object Detection
### GPU-Accelerated Autonomous Perception Pipeline on NVIDIA Jetson Orin Nano

![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin%20Nano-76B900?style=flat-square&logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Language](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20Python-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-TensorRT-FF6F00?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Overview

A full **real-time autonomous perception pipeline** built from scratch and deployed on embedded GPU hardware. The system combines:

- **Custom CUDA kernels** for Sobel-based lane/edge detection running at stable 30 fps
- **YOLOv8 object detection** converted to TensorRT for low-latency inference
- **GPU memory optimizations** benchmarked via NVIDIA Nsight profiling

Built for the *Parallel Programming (MENG 3540)* course at Humber Polytechnic — Winter 2026.

> 👩‍💻 **Author:** Tanvi Bhanderi — Mechatronics Engineering

---

## 🎬 Demo

| Lane / Edge Detection (CUDA Sobel) | YOLOv8 Live Object Detection |
|---|---|
| ![Edge Detection](Edge%20Detection.jpg) | ![YOLO Detection](YOLO%20Object%20Detection.jpeg) |

> Kernel Time: **0.646 ms** · Bandwidth: **0.95 GB/s** · Running live on Jetson Orin Nano

---

## ✨ Key Results

| Metric | Value |
|---|---|
| Pipeline frame rate | **30 fps (stable)** |
| CUDA kernel execution time | **~0.65 ms per frame** |
| CPU offload reduction | **~40%** vs. OpenCV CPU baseline |
| TensorRT inference speedup | **~35% latency reduction** vs. ONNX |
| GPU occupancy (optimized) | **~85%** |

---

## 🏗️ System Architecture

```
Live Camera Feed (USB)
        │
        ▼
  OpenCV Frame Capture
        │
   ┌────┴────────────────────────┐
   │                             │
   ▼                             ▼
CUDA Pipeline               YOLOv8 Branch
   │                             │
   ├─ RGB → Grayscale Kernel     ├─ ONNX → TensorRT conversion
   ├─ Sobel Gx / Gy Kernels      ├─ Real-time inference
   └─ Gradient Magnitude         └─ Bounding box overlay
        │                             │
        └────────────┬────────────────┘
                     ▼
              Display Output (imshow)
```

---

## 🔧 Technical Deep Dive

### 1. CUDA Sobel Edge Detection

Each video frame is processed entirely on the GPU. A custom `__global__` CUDA kernel applies the Sobel operator independently to every pixel — no CPU involvement during inference.

**Sobel Kernels (3×3):**
```
Gx (Horizontal):     Gy (Vertical):
-1   0  +1           -1  -2  -1
-2   0  +2            0   0   0
-1   0  +1           +1  +2  +1
```

**Gradient Magnitude:** `|Gx| + |Gy|` → Edge strength per pixel

**Thread mapping:** Each CUDA thread handles exactly one pixel.
Block size: `16×16` → Grid tiles cover the full frame resolution.

---

### 2. Memory Optimizations

Profiling with **NVIDIA Nsight** revealed the baseline kernel was **memory-bound** (~75% memory throughput, ~47% compute throughput). Two optimizations were implemented and compared:

| Version | Execution Time | Memory Throughput | Occupancy | Notes |
|---|---|---|---|---|
| Baseline (Global Memory) | ~124 µs | ~75% | ~83% | Memory bottleneck |
| Shared Memory | ~228 µs | ~48% | ~92% | Higher occupancy; overhead from tile loading |
| **Constant Memory** | **~134 µs** | **~76%** | **~85%** | ✅ Best overall — minimal overhead |

**Constant Memory** was selected as the final optimization — Sobel filter coefficients are stored in `__constant__` memory, reducing per-thread memory latency since all threads read identical coefficients.

---

### 3. YOLOv8 + TensorRT Object Detection

- YOLOv8 model exported from PyTorch → **ONNX → TensorRT** engine
- TensorRT INT8/FP16 quantization cut inference latency by **~35%**
- **CPU vs GPU comparison** scripts included — `cpu_live_yolo.py` vs `gpu_live_yolo.py` — with profiling benchmarks showing GPU advantage on Jetson hardware
- Live detection of vehicles, pedestrians, and common objects with confidence scores displayed per bounding box

---

## 📁 Repository Structure

```
📦 cuda-lane-object-detection/
├── 📁 src/
│   ├── step1_rgb_to_gray.cu       # CUDA grayscale kernel (camera feed)
│   ├── step2a_convolution.cu      # Basic 2D convolution kernel
│   ├── step2b_sobel_realtime.cu   # Sobel edge detection (live video)
│   ├── step2c_shared_memory.cu    # Optimization 1: Shared memory
│   └── step2c_constant_memory.cu  # Optimization 2: Constant memory ✅ Final
├── 📁 yolo/
│   ├── cpu_live_yolo.py           # YOLOv8 inference on CPU (baseline)
│   ├── gpu_live_yolo.py           # YOLOv8 inference on GPU (optimized)
│   └── yolo_test.py               # Model validation & test script
├── 📁 assets/
│   ├── Edge Detection.jpg             # Sobel output screenshot
│   ├── YOLO Object Detection.jpeg     # YOLOv8 bounding box screenshot
│   ├── cpu_usage.png              # CPU profiling benchmark
│   ├── gpu_usage.png              # GPU profiling benchmark
│   └── detection.png              # Detection results comparison
├── 📄 Project_Report.pdf
├── 📄 CMakeLists.txt
└── 📄 README.md
```

---

## 🚀 Getting Started

### Prerequisites

- NVIDIA Jetson Orin Nano (JetPack 6.2.1 / CUDA 12.6)
- OpenCV 4.x with CUDA support
- Python 3.8+ with `ultralytics`, `tensorrt`

### Build & Run (CUDA C++)

```bash
# Compile any step
nvcc -o lane_detect src/step2b_sobel_realtime.cu `pkg-config --cflags --libs opencv4` -O2

# Run with USB camera
./lane_detect
# Press ESC to exit
```

### Run YOLOv8 Detection

```bash
pip install ultralytics

# Run on CPU (baseline)
python yolo/cpu_live_yolo.py

# Run on GPU (optimized — recommended on Jetson)
python yolo/gpu_live_yolo.py

# Test model on static images
python yolo/yolo_test.py
```

### Convert Model to TensorRT (optional, for faster inference)

```bash
# Export YOLOv8 to TensorRT engine
yolo export model=yolov8n.pt format=engine device=0
```

---

## 🧠 What I Learned

- Writing and profiling **custom CUDA kernels** for real-time image processing
- Understanding GPU memory hierarchy: **global vs. shared vs. constant memory** — and when each matters
- Using **NVIDIA Nsight** to identify bottlenecks and validate optimizations with quantitative metrics
- Deploying a full **ML inference pipeline** on embedded hardware with TensorRT
- Balancing theoretical GPU occupancy vs. real-world execution time (shared memory added overhead despite higher occupancy)

---

## 📚 References

- NVIDIA CUDA Programming Guide
- Ultralytics YOLOv8 Documentation
- OpenCV CUDA Module Documentation
- Humber Polytechnic MENG 3540 — Parallel Programming, Winter 2026

---

<p align="center">
  Built by Tanvi Bhanderi · Humber Polytechnic — Mechatronics Engineering
</p>
