# 🎧 Whisper GPU API with FT Whisper + FastAPI + Docker

This project provides a FastAPI server that uses [ft-whspr]to transcribe audio files using GPU acceleration inside a Docker container.

---

## 🚀 Features

- FastAPI server with `/transcribe` endpoint
- Uses `` Whisper model
- GPU acceleration via CUDA 12.6
- Dockerized for portability

---

## 📁 Project Structure

```

whisper\_ft/
├── app/
│   └── main.py
├── requirements.txt
├── Dockerfile
└── README.md

````

---

## 🔧 Requirements

- **Ubuntu 24.04**
- **NVIDIA GPU** with driver installed
- **CUDA 12.6**
- **Docker**
- **NVIDIA Container Toolkit (with workaround for Ubuntu 24.04)**

---

## 🛠️ Setup Instructions

### 1. Clone this repo
```bash
git clone <your-repo-url>
cd whisper_ft
````

---

### 2. ✅ Fix NVIDIA Container Toolkit for Ubuntu 24.04

> NVIDIA's official Docker repos don't yet support Ubuntu 24.04, so we use 22.04 instead.

#### Clean up any broken setup:

```bash
sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt clean
sudo apt update
```

#### Install the NVIDIA container toolkit:

```bash
distribution="ubuntu22.04"

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### 3. 🧱 Build the Docker Image

```bash
docker build -t whisper_ft .
```

---

### 4. ▶️ Run the API Container with GPU

```bash
docker run --gpus all -p 8000:8000 whisper_ft
```

---

### 5. 🔁 Call the API

#### POST `/transcribe`

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H  "accept: application/json" \
  -H  "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```

---

## 📝 Example Response

```json
{
  "language": "en",
  "language_probability": 1.0,
  "transcript": "[0.00s -> 3.42s] Hello world, this is a test transcription.\n..."
}
```

---

## 📦 Python Dependencies

### `requirements.txt`

```
fastapi
uvicorn[standard]
faster-whisper
```

---

## 🐳 Dockerfile Notes

Uses:

```Dockerfile
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
```

Installs Python, FFmpeg, and your app.

---

## ❓Troubleshooting

* `could not select device driver "" with capabilities: [[gpu]]`
  → Install NVIDIA Container Toolkit as shown above.

* `libcudnn_ops.so not found`
  → Use CUDA base image with cuDNN preinstalled (`cudnn-runtime`).

---

## 📜 License

MIT or as per upstream project licenses.
