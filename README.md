# FLIoT 2nd Gen - Federated Learning for Semantic Segmentation

연합학습(Federated Learning)을 활용한 자율주행 환경의 시맨틱 세그멘테이션(Semantic Segmentation) 시스템입니다. SYNTHIA 데이터셋을 사용하여 분산된 클라이언트들 간의 협업 학습을 수행하며, oneM2M 표준 기반의 IoT 플랫폼(Mobius)을 통해 통신합니다.

## 목차

- [주요 특징](#주요-특징)
- [시스템 아키텍처](#시스템-아키텍처)
- [사전 요구사항](#사전-요구사항)
- [프로젝트 구조](#프로젝트-구조)
- [시작하기](#시작하기)
- [주요 컴포넌트](#주요-컴포넌트)
- [설정 가이드](#설정-가이드)

## 주요 특징

- **연합학습**: FedAvg 알고리즘 기반 분산 학습
- **Non-IID 데이터 분할**: Dirichlet 분포를 활용한 실제 환경 시뮬레이션
- **시맨틱 세그멘테이션**: DeepLabV3+ 및 U-Net 모델 지원
- **Knowledge Distillation**: 서버에서 Teacher 모델을 활용한 성능 향상
- **IoT 플랫폼 연동**: oneM2M 표준 기반 Mobius/nCube 통신
- **GPU 지원**: TensorFlow GPU 가속 및 자동 메모리 관리
- **실시간 모니터링**: 학습 메트릭 및 성능 지표 추적

## 시스템 아키텍처

```
┌─────────────┐     oneM2M      ┌─────────────┐
│   Client 1  │◄───────────────►│             │
├─────────────┤                 │   Mobius    │
│   Client 2  │◄───────────────►│    IoT      │
├─────────────┤                 │  Platform   │
│   Client 3  │◄───────────────►│             │
└─────────────┘                 └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │ FL Server   │
                                │ (FedAvg +   │
                                │     KD)     │
                                └─────────────┘
```

## 사전 요구사항

### 1. Docker 및 Docker Compose

```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Python 3.7+ 환경

```bash
# Python 및 pip 설치
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate
```

### 3. SYNTHIA 데이터셋

- [SYNTHIA 데이터셋](https://synthia-dataset.net/) 다운로드
- `Data/SYNTHIA/` 폴더에 압축 해제
- RGB 이미지 및 시맨틱 레이블 필요

## 프로젝트 구조

```
FLIoT2ndGen/
├── Client/                # 클라이언트 구현
│   ├── fl_client.py       # 메인 클라이언트 로직
│   ├── communication/     # TCP 통신 모듈
│   ├── federated/         # FL 프로토콜 핸들러
│   ├── model/             # 로컬 모델 및 데이터 로더
│   └── utils/             # GPU 설정, Pickle 유틸리티
├── Server/                # 서버 구현
│   ├── fl_server.py       # 메인 서버 로직
│   ├── communication/     # Mobius API 통신
│   ├── federated/         # FedAvg 집계 로직
│   ├── KD/                # Knowledge Distillation
│   ├── models/            # 글로벌 모델 정의
│   └── utils/             # GPU 설정, Pickle 유틸리티
├── Data/                  # 데이터셋 관리
│   ├── DataSplitter.py    # 데이터 분할 스크립트
│   ├── non_iid_split.py   # Non-IID 분할 로직
│   └── SYNTHIA/           # SYNTHIA 데이터셋 (사용자 추가)
├── Mobius/                # oneM2M IoT 플랫폼
│   └── mobius/            # Mobius 서버 코드
├── nCube/                 # IoT 게이트웨이
│   └── thyme.js           # nCube 메인 서버
├── template/              # Docker Compose 템플릿
├── envset/                # 하드웨어 초기 설정
├── results/               # 학습 결과 저장 (자동 생성)
├── .env.example           # 환경 변수 예제
├── docker-compose.yaml    # Docker Compose 설정
└── setup.sh               # 환경 설정 스크립트
```

## 시작하기

### 1. 데이터셋 준비

```bash
# SYNTHIA 데이터셋을 Data/SYNTHIA/ 폴더에 저장 후 실행
cd Data
python3 DataSplitter.py
```

이 스크립트는 다음을 수행합니다:
- `.env` 파일의 설정에 따라 데이터를 Non-IID로 분할
- 클라이언트별 train/test 데이터셋 생성
- `Server/SYNTHIA_Splitted/` 폴더에 분할된 데이터 저장
- 데이터 분포 시각화 그래프 생성

### 2. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 설정 조정
# - CLIENT: 클라이언트 수
# - LABEL: 사용할 클래스 (ALL 또는 인덱스)
# - DIRICHLET_ALPHA: Non-IID 강도 (작을수록 불균형)
# - RATIO: 클라이언트별 데이터 비율
# - TR_RATIO: 각 클라이언트의 train/test 비율

# setup.sh 실행 (선택사항)
./setup.sh
```

### 3. Docker Compose로 실행

```bash
# 컨테이너 빌드 및 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f

# 특정 서비스 로그 확인
docker-compose logs -f server
docker-compose logs -f client_1
```

실행 시 자동으로 수행되는 작업:
- **Mobius**: oneM2M IoT 플랫폼 시작
- **nCube**: IoT 게이트웨이 초기화 및 Mobius 연결
- **Server**: 글로벌 모델 초기화, AE/CNT 구독 설정
- **Clients**: 로컬 모델 로드, 서버와 통신 시작, 학습 시작

### 4. 종료

```bash
# 컨테이너 중지 및 제거
docker-compose down

# 볼륨까지 모두 제거
docker-compose down -v
```

## 주요 컴포넌트

### Client
- **역할**: 로컬 데이터로 모델 학습 및 서버에 가중치 전송
- **주요 기능**:
  - 로컬 모델 학습 (DeepLabV3+/U-Net)
  - TCP 기반 서버 통신
  - GPU/CPU 자동 감지 및 활용
  - 학습 메트릭 수집 및 전송
- 자세한 내용: [Client/README.md](Client/README.md)

### Server
- **역할**: 클라이언트 모델 집계 및 글로벌 모델 업데이트
- **주요 기능**:
  - FedAvg 알고리즘 구현
  - Knowledge Distillation (Teacher 모델)
  - Mobius IoT 플랫폼 연동
  - 학습 진행 상황 모니터링
- 자세한 내용: [Server/README.md](Server/README.md)

### Mobius
- **역할**: oneM2M 표준 기반 IoT 플랫폼
- **기능**: AE/CNT 관리, pub/sub 통신
- 자세한 내용: [Mobius/README.md](Mobius/README.md)

### nCube
- **역할**: IoT 게이트웨이
- **기능**: 디바이스와 Mobius 간 통신 중개
- 자세한 내용: [nCube/README.md](nCube/README.md)

## 설정 가이드

### 환경 변수 (.env)

```bash
# 네트워크 설정
DOCKER_NETWORK=172.20.0.

# 사용할 클래스 (ALL 또는 0~18 인덱스)
LABEL=ALL

# 입력 이미지 크기 (높이 너비 채널)
INPUT_SHAPE=432 768 3

# 데이터 셔플 강도 (0=원래 순서, 1=완전 랜덤)
SHUFFLE_INTENSITY=0

# Global Test 데이터 비율
GLOBAL_TEST_RATIO=0.1

# 서버용 데이터 비율
SERVER_RATIO=0.1

# 클라이언트 수
CLIENT=3

# 클라이언트별 데이터 비율
RATIO=1 1 1

# 각 클라이언트의 train/test 비율
TR_RATIO=0.8 0.8 0.8

# Dirichlet 알파 (작을수록 Non-IID 강도 증가)
DIRICHLET_ALPHA=0.1
```

### Docker Compose 설정

주요 서비스:
- `db`: MySQL 데이터베이스 (Mobius용)
- `mobius`: oneM2M IoT 플랫폼
- `nCube`: IoT 게이트웨이
- `server`: 연합학습 서버
- `client_1`, `client_2`, `client_3`: 연합학습 클라이언트

각 서비스는 `federated_network` (172.20.0.0/24)에 연결됩니다.

## 학습 결과

학습 결과는 `results/` 폴더에 자동 저장됩니다:
- 각 라운드별 모델 가중치
- 학습 메트릭 (loss, accuracy, IoU)
- 성능 지표 (CPU/GPU 사용량, 메모리, 학습 시간)

## 주요 알고리즘

### FedAvg (Federated Averaging)
- 클라이언트들의 로컬 모델 가중치를 데이터 크기 기반으로 가중 평균
- 글로벌 모델 업데이트

### Knowledge Distillation
- 서버에서 Teacher 모델 학습
- Teacher의 지식을 클라이언트에 전달하여 성능 향상

### Non-IID Data Split
- Dirichlet 분포를 활용한 불균형 데이터 분할
- 실제 환경의 데이터 이질성 시뮬레이션

## 문제 해결

### GPU 메모리 부족
```python
# Client/utils/gpu_setup.py 또는 Server/utils/gpu_setup.py에서
# GPU 메모리 성장 허용 설정 확인
```

### Mobius 연결 실패
```bash
# Mobius 컨테이너 상태 확인
docker-compose logs mobius

# DB 헬스체크 확인
docker-compose ps db
```

### 데이터 로드 오류
```bash
# 데이터 분할이 올바르게 되었는지 확인
ls -la Data/SYNTHIA/
ls -la Server/SYNTHIA_Splitted/
```

## 참고 문헌

- [Federated Learning](https://arxiv.org/abs/1602.05629)
- [SYNTHIA Dataset](https://synthia-dataset.net/)
- [oneM2M Standard](https://www.onem2m.org/)
- [DeepLabV3+](https://arxiv.org/abs/1802.02611)

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.

## 기여

버그 리포트 및 개선 제안은 Issue를 통해 제출해주세요.
