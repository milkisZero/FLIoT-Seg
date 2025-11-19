# CUDA 11.8 + cuDNN8 (Ubuntu 22.04)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 3.10 & 빌드 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

# python/pip 편의 링크
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "fl_client.py"]
