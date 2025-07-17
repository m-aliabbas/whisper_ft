FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Install system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY app /app

# Expose app port
EXPOSE 8000

# Set default number of workers
ENV UVICORN_WORKERS=1

# Run the app using the configurable number of workers
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8000 --workers $UVICORN_WORKERS"]
