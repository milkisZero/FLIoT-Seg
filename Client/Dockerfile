FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get update && apt-get install -y \
build-essential \
gcc \
g++ \
gfortran \
libopenblas-dev \
liblapack-dev \
pkg-config \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "fl_client.py"]
