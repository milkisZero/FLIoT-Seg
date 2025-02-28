FROM python:3.7.12-slim

RUN apt-get update && apt-get install -y \
build-essential \
gcc \
g++ \
gfortran \
libopenblas-dev \
liblapack-dev \
pkg-config \
curl \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "fl_server.py"]