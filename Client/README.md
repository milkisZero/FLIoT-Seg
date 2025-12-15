# Client - Federated Learning Client

연합학습 시스템의 클라이언트 구현입니다. 로컬 데이터로 시맨틱 세그멘테이션 모델을 학습하고, 학습된 가중치를 서버에 전송합니다.

## 목차

- [개요](#개요)
- [프로젝트 구조](#프로젝트-구조)
- [주요 기능](#주요-기능)
- [설치 및 실행](#설치-및-실행)
- [구성 요소](#구성-요소)
- [통신 프로토콜](#통신-프로토콜)

## 개요

클라이언트는 다음과 같은 역할을 수행합니다:
1. 로컬 데이터셋으로 모델 학습
2. 서버로부터 글로벌 모델 수신
3. 로컬 모델 가중치를 서버에 전송
4. 학습 메트릭 수집 및 보고

## 프로젝트 구조

```
Client/
├── fl_client.py              # 메인 클라이언트 실행 파일
├── communication/            # 통신 모듈
│   └── tcp_client.py         # TCP 소켓 클라이언트
├── federated/                # 연합학습 관련 모듈
│   ├── protocol_handler.py   # FL 프로토콜 핸들러
│   └── result_manager.py     # 결과 저장 및 관리
├── model/                    # 모델 관련 모듈
│   ├── local_model.py        # 로컬 모델 정의
│   └── datasetLoader.py      # 데이터셋 로더
├── generator/                # 데이터 제너레이터
├── utils/                    # 유틸리티 모듈
│   ├── gpu_setup.py          # GPU 설정
│   └── pickle_utils.py       # Pickle 직렬화
├── Dockerfile                # Docker 이미지 설정
└── requirements.txt          # Python 패키지 의존성
```

## 주요 기능

### 1. 로컬 모델 학습
- **모델 아키텍처**: DeepLabV3+ (MobileNetV2 백본) 또는 U-Net
- **학습 방식**:
  - 로컬 데이터셋으로 지도 학습
  - 표준 연합학습 (FedAvg)
  - GPU/CPU 자동 감지 및 활용
- **최적화**: Adam optimizer, categorical crossentropy loss

### 2. 서버 통신
- **프로토콜**: TCP 소켓 기반
- **통신 내용**:
  - 글로벌 모델 가중치 수신
  - 로컬 모델 가중치 전송
  - 학습 메트릭 전송 (loss, accuracy, IoU)

### 3. 데이터 처리
- **데이터셋**: SYNTHIA (자율주행 시맨틱 세그멘테이션)
- **전처리**:
  - 이미지 리사이징 (432x768)
  - 정규화 (0~1 범위)
  - 레이블 인코딩 (19 클래스)
- **Non-IID 지원**: Dirichlet 분포 기반 데이터 분할

### 4. 성능 모니터링
- CPU/GPU 사용량 추적
- 메모리 사용량 측정
- 학습 시간 기록
- 세그멘테이션 성능 지표 (IoU, mIoU)

## 설치 및 실행

### 로컬 환경에서 실행

```bash
# 의존성 설치
cd Client
pip install -r requirements.txt

# 환경 변수 설정
export CLIENT_ID=1
export SERVER_HOST=172.20.0.100
export SERVER_PORT=5000

# 클라이언트 실행
python3 fl_client.py
```

### Docker로 실행

```bash
# Docker 이미지 빌드
docker build -t fl-client:latest .

# 컨테이너 실행
docker run -d \
  --name client_1 \
  -e CLIENT_ID=1 \
  -v $(pwd)/Data:/app/Data \
  fl-client:latest
```

### Docker Compose로 실행 (권장)

```bash
# 프로젝트 루트에서
docker-compose up -d client_1 client_2 client_3
```

## 구성 요소

### fl_client.py
메인 클라이언트 로직:
```python
# 주요 기능
1. 로컬 모델 초기화
2. 데이터 로더 설정
3. 서버와 연결
4. 연합학습 라운드 실행:
   - 글로벌 모델 수신
   - 로컬 학습 수행
   - 가중치 전송
   - 메트릭 보고
```

### communication/tcp_client.py
TCP 소켓 통신:
```python
# 기능
- 서버 연결 관리
- 데이터 송수신
- 재연결 로직
- 타임아웃 처리
```

### federated/protocol_handler.py
연합학습 프로토콜:
```python
# 기능
- 메시지 직렬화/역직렬화
- 프로토콜 버전 관리
- 에러 핸들링
- 상태 관리
```

### model/local_model.py
로컬 모델 정의:
```python
# 지원 모델
1. DeepLabV3+ (MobileNetV2)
   - 경량화된 시맨틱 세그멘테이션
   - IoT 디바이스에 적합

2. U-Net
   - 고성능 세그멘테이션
   - 더 많은 메모리 요구
```

### model/datasetLoader.py
데이터셋 로더:
```python
# 기능
- SYNTHIA 데이터셋 로드
- 배치 생성
- 데이터 증강 (선택사항)
- 메모리 효율적 처리
```

### utils/gpu_setup.py
GPU 설정:
```python
# 기능
- GPU 가용성 확인
- 메모리 성장 허용 설정
- 다중 GPU 지원
- TensorFlow GPU 최적화
```

## 통신 프로토콜

### 1. 초기 연결
```
Client -> Server: CONNECT (client_id, capabilities)
Server -> Client: ACCEPT (session_id, config)
```

### 2. 학습 라운드
```
Server -> Client: START_ROUND (round_num, global_weights)
Client: 로컬 학습 수행
Client -> Server: WEIGHTS_UPDATE (round_num, local_weights, metrics)
Server -> Client: ROUND_COMPLETE (aggregated_weights)
```

### 3. 종료
```
Client -> Server: DISCONNECT
Server -> Client: BYE
```

## 환경 변수

클라이언트는 다음 환경 변수를 사용합니다:

```bash
# 필수 환경 변수
CLIENT_ID=1                    # 클라이언트 ID (1, 2, 3, ...)
SERVER_HOST=172.20.0.100       # 서버 IP 주소
SERVER_PORT=5000               # 서버 포트

# 선택적 환경 변수
BATCH_SIZE=4                   # 배치 크기
EPOCHS=1                       # 로컬 에폭 수
LEARNING_RATE=0.001            # 학습률
GPU_MEMORY_LIMIT=4096          # GPU 메모리 제한 (MB)
```

## 데이터셋 경로

클라이언트는 다음 경로에서 데이터를 로드합니다:

```
Server/SYNTHIA_Splitted/
├── client1_train/             # 클라이언트 1 학습 데이터
│   ├── RGB/                   # 입력 이미지
│   └── GT/                    # Ground Truth 레이블
├── client1_test/              # 클라이언트 1 테스트 데이터
├── client2_train/             # 클라이언트 2 학습 데이터
├── client2_test/
├── client3_train/             # 클라이언트 3 학습 데이터
└── client3_test/
```

## 학습 메트릭

클라이언트는 다음 메트릭을 수집합니다:

### 모델 성능
- **Loss**: Categorical crossentropy
- **Accuracy**: 픽셀 단위 정확도
- **IoU**: Intersection over Union (클래스별)
- **mIoU**: Mean IoU (전체 클래스 평균)

### 시스템 성능
- **CPU 사용량**: psutil 기반 측정
- **GPU 사용량**: CUDA 메모리 사용량
- **메모리 사용량**: RAM 사용량
- **학습 시간**: 에폭당/라운드당 시간

## 로깅

클라이언트는 다음 정보를 로깅합니다:

```python
# 로그 레벨
INFO: 일반 정보 (연결, 라운드 시작/종료)
DEBUG: 디버깅 정보 (데이터 크기, 가중치 shape)
WARNING: 경고 (재연결, 성능 저하)
ERROR: 에러 (연결 실패, 학습 실패)
```

로그는 표준 출력 및 `results/client_{id}/logs/` 폴더에 저장됩니다.

## 문제 해결

### GPU 메모리 부족
```python
# utils/gpu_setup.py 수정
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # 또는 메모리 제한 설정
    tf.config.set_logical_device_configuration(
        gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )
```

### 서버 연결 실패
```bash
# 네트워크 확인
docker network inspect federated_network

# 서버 상태 확인
docker-compose logs server

# 방화벽 설정 확인
sudo ufw status
```

### 데이터 로드 오류
```bash
# 데이터 경로 확인
ls -la Server/SYNTHIA_Splitted/client1_train/

# 권한 확인
chmod -R 755 Server/SYNTHIA_Splitted/
```

## 성능 최적화

### 1. 배치 크기 조정
```python
# 메모리가 충분한 경우
BATCH_SIZE = 8

# 메모리가 제한적인 경우
BATCH_SIZE = 2
```

### 2. Mixed Precision 학습
```python
# fl_client.py에서
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### 3. 데이터 캐싱
```python
# datasetLoader.py에서
dataset = dataset.cache()  # 메모리에 캐싱
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## 의존성

주요 패키지 ([requirements.txt](requirements.txt) 참조):

```
tensorflow==2.10.0           # 딥러닝 프레임워크
numpy==1.21.6                # 수치 연산
pandas==1.3.5                # 데이터 처리
scikit-learn==1.0.2          # 머신러닝 유틸
matplotlib==3.5.1            # 시각화
Pillow==9.5.0                # 이미지 처리
psutil                       # 시스템 모니터링
msgpack==1.0.3               # 직렬화
Flask-SocketIO==4.3.1        # 실시간 통신
```

## 참고 자료

- [DeepLabV3+ 논문](https://arxiv.org/abs/1802.02611)
- [U-Net 논문](https://arxiv.org/abs/1505.04597)
- [SYNTHIA 데이터셋](https://synthia-dataset.net/)
- [TensorFlow 문서](https://www.tensorflow.org/api_docs)

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.
