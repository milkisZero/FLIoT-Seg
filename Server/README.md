# Server - Federated Learning Server

연합학습 시스템의 중앙 서버 구현입니다. 클라이언트들로부터 모델 가중치를 수집하고 FedAvg 알고리즘으로 집계하여 글로벌 모델을 업데이트합니다.

## 목차

- [개요](#개요)
- [프로젝트 구조](#프로젝트-구조)
- [주요 기능](#주요-기능)
- [설치 및 실행](#설치-및-실행)
- [구성 요소](#구성-요소)
- [연합학습 알고리즘](#연합학습-알고리즘)
- [설정](#설정)

## 개요

서버는 다음과 같은 역할을 수행합니다:
1. 글로벌 모델 초기화 및 관리
2. 클라이언트 모델 가중치 집계 (FedAvg)
3. oneM2M 표준 기반 IoT 플랫폼 통신
4. Knowledge Distillation을 위한 Teacher 모델 학습
5. 학습 진행 상황 모니터링 및 로깅

## 프로젝트 구조

```
Server/
├── fl_server.py                 # 메인 서버 실행 파일
├── communication/               # 통신 모듈
│   ├── mobius_handler.py        # Mobius 플랫폼 핸들러
│   ├── mobius_routes.py         # Mobius REST API 라우트
│   └── api_routes.py            # Flask API 라우트
├── federated/                   # 연합학습 모듈
│   ├── protocol_handler.py      # FL 프로토콜 핸들러
│   └── result_manager.py        # 결과 저장 및 관리
├── KD/                          # Knowledge Distillation
│   ├── kd_server.py             # KD 서버 로직
│   ├── kd_handler.py            # KD 핸들러
│   └── teacher_pipeline.py      # Teacher 모델 파이프라인
├── models/                      # 모델 정의
│   ├── base_model.py            # 기본 모델 클래스
│   ├── DeepLabV3PlusMobileNet.py # DeepLabV3+ 모델
│   ├── unet_model.py            # U-Net 모델
│   └── datasetLoader.py         # 데이터셋 로더
├── utils/                       # 유틸리티 모듈
│   ├── gpu_setup.py             # GPU 설정
│   └── pickle_utils.py          # Pickle 직렬화
├── SYNTHIA_Splitted/            # 분할된 데이터셋 (자동 생성)
├── config.json                  # 서버 설정 파일
├── Dockerfile                   # Docker 이미지 설정
└── requirements.txt             # Python 패키지 의존성
```

## 주요 기능

### 1. 연합학습 오케스트레이션
- **FedAvg 알고리즘**: 클라이언트 모델 가중치를 데이터 크기 기반 가중 평균
- **라운드 관리**: 학습 라운드 시작/종료 제어
- **클라이언트 동기화**: 모든 클라이언트 응답 대기 및 집계
- **글로벌 모델 배포**: 업데이트된 모델을 클라이언트에 전송

### 2. IoT 플랫폼 연동
- **Mobius 통신**: oneM2M 표준 기반 pub/sub
- **AE/CNT 관리**: Application Entity 및 Container 구독
- **실시간 통신**: MQTT/HTTP 기반 양방향 통신
- **자동 재연결**: 연결 실패 시 자동 복구

### 3. Knowledge Distillation
- **Teacher 모델**: 서버 데이터로 고성능 모델 학습
- **지식 전달**: Teacher의 soft label을 클라이언트에 제공
- **성능 향상**: 클라이언트 모델의 일반화 성능 개선
- **경량화 지원**: 작은 모델도 높은 성능 달성

### 4. 모니터링 및 로깅
- 각 라운드별 메트릭 수집
- 클라이언트별 성능 추적
- 글로벌 모델 성능 평가
- 시스템 리소스 모니터링

## 설치 및 실행

### 로컬 환경에서 실행

```bash
# 의존성 설치
cd Server
pip install -r requirements.txt

# 설정 파일 확인
cat config.json

# 서버 실행
python3 fl_server.py
```

### Docker로 실행

```bash
# Docker 이미지 빌드
docker build -t fl-server:latest .

# 컨테이너 실행
docker run -d \
  --name server \
  -p 5000:5000 \
  -v $(pwd)/results:/app/results \
  fl-server:latest
```

### Docker Compose로 실행 (권장)

```bash
# 프로젝트 루트에서
docker-compose up -d server
```

## 구성 요소

### fl_server.py
메인 서버 로직:
```python
# 주요 기능
1. Flask 웹 서버 초기화
2. Mobius IoT 플랫폼 연결
3. 글로벌 모델 초기화
4. 연합학습 라운드 실행:
   - 클라이언트 모델 수집
   - FedAvg 집계
   - 글로벌 모델 업데이트
   - 결과 저장
5. Knowledge Distillation (선택사항)
```

### communication/mobius_handler.py
Mobius IoT 플랫폼 통신:
```python
# 기능
- AE(Application Entity) 등록
- Container 생성 및 구독
- ContentInstance 생성/조회
- Notification 수신 처리
- 재연결 및 에러 복구
```

### communication/mobius_routes.py
Mobius REST API:
```python
# 엔드포인트
POST /api/mobius/notification    # Mobius 알림 수신
GET  /api/mobius/status           # Mobius 연결 상태
POST /api/mobius/publish          # 데이터 발행
```

### communication/api_routes.py
클라이언트 API:
```python
# 엔드포인트
POST /api/client/register         # 클라이언트 등록
POST /api/client/weights          # 모델 가중치 전송
GET  /api/client/global_model     # 글로벌 모델 조회
POST /api/client/metrics          # 메트릭 전송
```

### federated/protocol_handler.py
연합학습 프로토콜:
```python
# 기능
- FedAvg 알고리즘 구현
- 가중치 집계 로직
- 클라이언트 선택 (선택사항)
- 라운드 상태 관리
```

FedAvg 구현:
```python
def aggregate_weights(client_weights, client_sizes):
    """
    클라이언트 가중치를 데이터 크기 기반으로 집계

    Args:
        client_weights: 클라이언트별 모델 가중치 리스트
        client_sizes: 클라이언트별 데이터 크기 리스트

    Returns:
        aggregated_weights: 집계된 글로벌 모델 가중치
    """
    total_size = sum(client_sizes)
    aggregated = []

    for layer_idx in range(len(client_weights[0])):
        weighted_sum = sum(
            weight[layer_idx] * (size / total_size)
            for weight, size in zip(client_weights, client_sizes)
        )
        aggregated.append(weighted_sum)

    return aggregated
```

### KD/teacher_pipeline.py
Knowledge Distillation 파이프라인:
```python
# 기능
1. Teacher 모델 초기화 (더 큰 모델)
2. 서버 데이터로 Teacher 학습
3. Soft label 생성
4. Student(클라이언트) 모델에 전달
5. Distillation loss 계산
```

### models/DeepLabV3PlusMobileNet.py
DeepLabV3+ 모델:
```python
# 특징
- MobileNetV2 백본 (경량화)
- Atrous Spatial Pyramid Pooling (ASPP)
- Decoder with skip connections
- 19개 클래스 시맨틱 세그멘테이션
```

### models/unet_model.py
U-Net 모델:
```python
# 특징
- Encoder-Decoder 아키텍처
- Skip connections
- 높은 정확도
- 더 많은 메모리 요구
```

## 연합학습 알고리즘

### FedAvg (Federated Averaging)

```
알고리즘: FedAvg
입력:
  - K: 클라이언트 수
  - T: 총 라운드 수
  - η: 학습률
  - B: 배치 크기
  - E: 로컬 에폭 수

서버 실행:
  1. 글로벌 모델 w₀ 초기화

  2. for each round t = 1 to T:
     a. 클라이언트 선택: St ⊆ {1, ..., K}
     b. for each client k ∈ St in parallel:
        - wₖᵗ⁺¹ ← ClientUpdate(k, wᵗ)
     c. 가중치 집계:
        wᵗ⁺¹ ← Σₖ (nₖ/n) * wₖᵗ⁺¹
        (nₖ: 클라이언트 k의 데이터 크기, n: 전체 데이터 크기)

ClientUpdate(k, w):
  1. 로컬 모델에 w 로드
  2. for each local epoch i = 1 to E:
     - for each batch b:
       w ← w - η∇ℓ(w; b)
  3. return w
```

### Knowledge Distillation (선택사항)

```
Teacher 모델 학습:
  1. 서버 데이터로 고성능 모델 학습
  2. Soft label 생성: p_soft = softmax(logits / T)
     (T: temperature parameter)

Student 모델 학습:
  1. Hard label loss: L_hard = CE(y, y_pred)
  2. Soft label loss: L_soft = KL(p_soft, p_student)
  3. Total loss: L = αL_hard + (1-α)L_soft
```

## 설정

### config.json
서버 설정 파일:
```json
{
  "model": {
    "type": "deeplabv3plus",
    "input_shape": [432, 768, 3],
    "num_classes": 19,
    "backbone": "mobilenetv2"
  },
  "training": {
    "rounds": 100,
    "local_epochs": 1,
    "batch_size": 4,
    "learning_rate": 0.001
  },
  "federated": {
    "min_clients": 3,
    "client_fraction": 1.0,
    "aggregation": "fedavg"
  },
  "kd": {
    "enabled": true,
    "temperature": 3.0,
    "alpha": 0.7,
    "teacher_model": "deeplabv3plus_large"
  },
  "mobius": {
    "host": "172.20.0.5",
    "port": 7579,
    "ae_name": "FLServer",
    "cse_base": "/Mobius"
  }
}
```

### 환경 변수

```bash
# Flask 설정
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false

# Mobius 설정
MOBIUS_HOST=172.20.0.5
MOBIUS_PORT=7579

# 학습 설정
FL_ROUNDS=100
MIN_CLIENTS=3
BATCH_SIZE=4

# GPU 설정
GPU_MEMORY_LIMIT=8192
CUDA_VISIBLE_DEVICES=0
```

## 통신 프로토콜

### oneM2M 구조

```
Mobius (CSE)
└── FLServer (AE)
    ├── global_model (CNT)
    │   └── weights_v1 (CIN)
    ├── client_1 (CNT)
    │   └── local_weights (CIN)
    ├── client_2 (CNT)
    │   └── local_weights (CIN)
    └── client_3 (CNT)
        └── local_weights (CIN)
```

### 학습 플로우

```
1. 초기화
   Server -> Mobius: AE 등록 (FLServer)
   Server -> Mobius: CNT 생성 (global_model)

2. 클라이언트 등록
   Client -> Server: 등록 요청
   Server -> Mobius: CNT 생성 (client_X)
   Server -> Mobius: CNT 구독 설정

3. 학습 라운드
   Server -> Mobius: 글로벌 모델 발행 (CIN)
   Clients: 글로벌 모델 조회
   Clients: 로컬 학습
   Clients -> Mobius: 로컬 가중치 발행 (CIN)
   Server: Notification 수신
   Server: FedAvg 집계
   Server: 글로벌 모델 업데이트
```

## API 엔드포인트

### 클라이언트 API

```
POST /api/client/register
Body: {
  "client_id": "client_1",
  "data_size": 1000
}
Response: {
  "status": "registered",
  "session_id": "abc123"
}

POST /api/client/weights
Body: {
  "client_id": "client_1",
  "round": 1,
  "weights": <serialized weights>,
  "data_size": 1000
}
Response: {
  "status": "received"
}

GET /api/client/global_model?round=1
Response: {
  "round": 1,
  "weights": <serialized weights>
}

POST /api/client/metrics
Body: {
  "client_id": "client_1",
  "round": 1,
  "metrics": {
    "loss": 0.5,
    "accuracy": 0.85,
    "iou": 0.72
  }
}
Response: {
  "status": "saved"
}
```

### Mobius API

```
POST /api/mobius/notification
Body: <oneM2M notification>
Response: {
  "status": "processed"
}

GET /api/mobius/status
Response: {
  "connected": true,
  "ae_registered": true,
  "subscriptions": 3
}
```

## 학습 결과

결과는 `results/` 폴더에 저장됩니다:

```
results/
├── global_model/
│   ├── round_1.h5
│   ├── round_2.h5
│   └── ...
├── metrics/
│   ├── server_metrics.json
│   ├── client_1_metrics.json
│   ├── client_2_metrics.json
│   └── client_3_metrics.json
└── logs/
    └── server.log
```

### 메트릭 구조

```json
{
  "round": 1,
  "timestamp": "2025-12-15T10:30:00",
  "global_metrics": {
    "loss": 0.45,
    "accuracy": 0.87,
    "mean_iou": 0.75
  },
  "client_metrics": {
    "client_1": {
      "loss": 0.48,
      "accuracy": 0.85,
      "data_size": 1000
    },
    "client_2": { ... },
    "client_3": { ... }
  },
  "aggregation_time": 2.5,
  "total_clients": 3
}
```

## 문제 해결

### Mobius 연결 실패
```bash
# Mobius 컨테이너 확인
docker-compose logs mobius

# DB 연결 확인
docker-compose exec db mysql -u root -p -e "SHOW DATABASES;"

# 네트워크 확인
docker network inspect federated_network
```

### 클라이언트 응답 없음
```bash
# 클라이언트 상태 확인
docker-compose logs client_1

# 서버 로그 확인
docker-compose logs server

# 타임아웃 설정 조정 (config.json)
{
  "timeout": {
    "client_response": 300,
    "round_timeout": 600
  }
}
```

### GPU 메모리 부족
```python
# utils/gpu_setup.py에서
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### FedAvg 집계 오류
```python
# 가중치 shape 불일치 확인
# 모든 클라이언트가 동일한 모델 아키텍처 사용하는지 확인
```

## 성능 최적화

### 1. 클라이언트 선택
```python
# 모든 클라이언트 대신 일부만 선택 (빠른 학습)
client_fraction = 0.5  # 50%의 클라이언트만 선택
```

### 2. 비동기 집계
```python
# 느린 클라이언트 대기 없이 집계
async_aggregation = True
min_clients_for_aggregation = 2  # 최소 2개 클라이언트
```

### 3. 압축
```python
# 가중치 압축으로 통신 비용 감소
compression = "quantization"  # 또는 "sparsification"
```

## 의존성

주요 패키지 ([requirements.txt](requirements.txt) 참조):

```
tensorflow==2.10.0           # 딥러닝 프레임워크
Flask==2.1.1                 # 웹 서버
Flask-SocketIO==4.3.1        # 실시간 통신
numpy==1.21.6                # 수치 연산
pandas==1.3.5                # 데이터 처리
requests==2.31.0             # HTTP 클라이언트
psutil                       # 시스템 모니터링
msgpack==1.0.3               # 직렬화
```

## 참고 자료

- [Federated Learning 논문](https://arxiv.org/abs/1602.05629)
- [oneM2M 표준](https://www.onem2m.org/)
- [Knowledge Distillation 논문](https://arxiv.org/abs/1503.02531)
- [DeepLabV3+ 논문](https://arxiv.org/abs/1802.02611)

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.
