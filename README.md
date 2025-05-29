# FLIoT - Federated Learning for IoT System

이 프로젝트는 IoT 플랫폼을 이용한 사이버 공격 탐지 모델 연합학습(Federated Learning) 시스템입니다. CIC-IDS2017 데이터셋을 활용하여 분산된 클라이언트들 간의 협업 학습을 도커로 구현하여 누구나 쉽게 테스트 해 봄을 목적으로 합니다.

## 목차
- [프로젝트 구조](#프로젝트-구조)
- [시작하기](#시작하기)
  - [CICIDS2017 데이터셋 다운로드](#1-cicids2017-데이터셋-다운로드)
  - [환경 설정](#2-환경-설정)
  - [실행](#3-실행)
  - [학습 메트릭 시각화](#4-학습-메트릭-시각화)
  - [공격자 패킷 구현](#5-공격자-패킷-구현)
- [주요 컴포넌트](#주요-컴포넌트)
- [유틸리티 도구](#유틸리티-도구)
- [요구사항](#요구사항)

## 프로젝트 구조

```
FLIoT
├── 📁 Client/              # 클라이언트 관련 코드
├── 📁 Server/              # 서버 관련 코드
├── 📁 Mobius/              # Mobius 플랫폼 관련 코드
├── 📁 nCube/               # nCube 관련 코드
├── 📁 template/            # 도커 템플릿 파일
├── 📁 envset/              # 하드웨어 초기 설정 파일
│   ├── 📁 forjetinit/      # Jetson 초기 설정
│   └── 📁 forraspinit/     # Raspberry Pi 초기 설정
├── 📁 results/             # 학습 결과 저장 폴더 (도커 실행 시 자동 생성)
├── 📁 graphs/              # 학습 메트릭 그래프 저장 폴더 (makegraph.py 실행 시 자동 생성)
├── .env.example            # 환경 변수 예제 파일
├── setup.sh                # 도커 환경 설정 스크립트
├── makegraph.py            # 학습 메트릭 시각화 도구
└── label_csv_sender.py     # 데이터 셋 기반 공격자 코드
```

> 📁 기본 폴더

## 시작하기

### 1. CICIDS2017 데이터셋 다운로드
   - [CICIDS2017 공식 다운로드 페이지](https://www.unb.ca/cic/datasets/ids-2017.html)에서 데이터셋을 다운로드
   - 다운로드한 CSV 파일들을 `Client/CICIDS/` 폴더에 저장

### 2. 환경 설정
   ```bash
   # setup.sh 실행
   ./setup.sh
   ```
   
   setup.sh 스크립트는 다음과 같은 자동화 작업을 수행합니다:
   - .env 파일이 없는 경우 .env.example을 자동으로 .env로 복사
   - template/ 폴더의 템플릿을 기반으로 docker-compose.yaml 파일 생성
   - .env 파일의 설정을 기반으로 도커 네트워크 구성
   - DatasetPreprocessCICIDS2017.py를 자동으로 실행하여 Client/CICIDS_Splitted/ 폴더에 데이터셋 분할

### 3. 실행
   ```bash
   # 도커 컴포즈로 실행
   docker-compose up -d --build
   ```
   
   실행 시 자동으로 수행되는 작업:
   - 서버
     - Mobius IoT 플랫폼 연결
     - AE(AE) 감지 및 구독 설정
     - 컨테이너(CNT) 구독 설정
     - 글로벌 모델 초기화
   - 클라이언트
     - Gateway 서버와 TCP 소켓 통신
     - 로컬 모델 학습 및 평가
     - 실시간 패킷 분류
     - 학습 결과 및 메트릭 전송
   
   종료 방법:
   ```bash
   # 컨테이너와 볼륨 모두 제거
   docker-compose down -v
   ```
   
   실행 시 자동으로 생성되는 폴더:
   - `results/`: 각 클라이언트의 학습 결과와 메트릭이 저장되는 폴더
     - CPU/GPU 사용량
     - 학습 시간
     - 메모리 사용량 등의 정보가 JSON 형식으로 저장

### 4. 학습 메트릭 시각화
   ```bash
   # 필요한 패키지 설치
   pip3 install matplotlib pandas numpy seaborn argparse
   
   # 그래프 생성
   python3 makegraph.py results/
   ```
   
   makegraph.py 실행 시 자동으로 생성되는 폴더:
   - `graphs/`: 학습 메트릭 시각화 결과가 저장되는 폴더
     - CPU/GPU 사용량 그래프
     - 학습 시간 그래프
     - 메모리 사용량 그래프가 PNG 형식으로 저장

### 5. 공격자 패킷 구현
   ```bash
   # 필요한 패키지 설치
   pip3 install pandas numpy scikit-learn python-dotenv
   
   # 자동 모드로 실행 (환경 변수 사용)
   python3 label_csv_sender.py --auto
   
   # 수동 모드로 실행
   python3 label_csv_sender.py
   ```
   
   label_csv_sender.py는 다음과 같은 기능을 수행합니다:
   - CICIDS2017 데이터셋을 기반으로 공격자 패킷 생성
   - 선택된 공격 유형의 패킷을 클라이언트로 전송
   - TCP 소켓을 통한 실시간 패킷 전송
   
   실행 시 필요한 환경 변수 (.env 파일):
   - LABEL: 선택할 공격 유형 인덱스 (공백으로 구분)
   - CLIENT: 공격 대상 클라이언트 수

## 주요 컴포넌트

- **Client**: 연합학습에 참여하는 개별 클라이언트 구현
  - 서버와의 통신 처리
    - Gateway 서버와 TCP 소켓 통신
    - 모델 가중치 및 학습 결과 전송
    - 실시간 학습 메트릭 전송 (loss, accuracy 등)
    - Gateway를 통한 중앙 서버와의 통신
  - 공격자 패킷 처리
    - 공격자 클라이언트의 TCP 연결 수신
    - 실시간 패킷 분류 및 결과 전송
    - 분류 결과 통계 수집 및 전송
  - 로컬 모델 학습
    - TensorFlow/Keras 기반 딥러닝 모델
    - GPU/CPU 자동 감지 및 설정
    - 학습 메트릭 수집 (시간, 메모리 사용량)
  - 모델 평가 및 분류
    - 실시간 패킷 분류 (normal/anomaly)
    - 분류 결과 통계 수집
    - F1-score, Precision, Recall 계산
  - 자동화된 기능
    - 학습 결과 자동 저장 (JSON 형식)
    - GPU 메모리 자동 관리
    - 오류 복구 및 재연결
  - 자세한 내용은 [Client/README.md](Client/README.md) 참조
- **Server**: 중앙 서버 구현
  - Mobius IoT 플랫폼 연동
    - AE(AE) 감지 및 구독 (aeWatcher)
    - 컨테이너(CNT) 구독 및 관리
    - 클라이언트와의 pub/sub 통신
  - 연합학습 관리
    - FedAvg 알고리즘 구현
    - 글로벌 모델 가중치 집계
    - 클라이언트 모델 업데이트
  - 학습 메트릭 수집
    - 손실 함수 값 및 정확도
    - CPU/GPU 사용량
    - 메모리 사용량
  - 자세한 내용은 [Server/README.md](Server/README.md) 참조
- **Mobius**: IoT 플랫폼 연동
  - `conf.json`: Mobius 서버 설정 파일
    - IP 주소 및 포트 설정
    - MQTT, CoAP, WebSocket 프로토콜 설정
  - `pxy_mqtt.js`: MQTT 프록시 설정
  - `pxy_coap.js`: CoAP 프록시 설정
  - `pxy_ws.js`: WebSocket 프록시 설정
  - 자세한 내용은 [Mobius/README.md](Mobius/README.md) 참조
- **nCube**: IoT 게이트웨이 구현
  - `conf.js`: nCube 서버 설정 파일
    - IP 주소 및 포트 설정
    - MQTT 브로커 설정
  - `onem2m_client.js`: oneM2M 클라이언트 구현
  - `thyme_tas.js`: Thing Adaptation Software 구현
  - `tas_emulator_FL2.js`: TAS 에뮬레이터
  - `thyme.js`: nCube 메인 서버
    - oneM2M 표준 기반 IoT 디바이스 구현
    - Mobius 서버와의 통신 처리
    - IP 변경 시 conf.js의 설정 확인 필요
  - 자세한 내용은 [nCube/README.md](nCube/README.md) 참조
- **envset**: 도커 외 하드웨어 초기 설정
  - `forjetinit/`: Jetson 보드 초기 설정
    - Jetson 보드의 기본 환경 설정
    - CUDA 및 TensorRT 설치 및 설정
    - 필요한 패키지 설치 (Python, pip, numpy 등)
  - `forraspinit/`: Raspberry Pi 초기 설정
    - Raspberry Pi의 기본 환경 설정
    - 필요한 패키지 설치 (Python, pip, RPi.GPIO 등)

## 유틸리티 도구

- **setup.sh**: 도커 환경 설정 스크립트
  - .env 파일의 설정을 기반으로 도커 컴포즈 파일 생성
  - 네트워크 설정 및 컨테이너 구성
  - 환경 변수 자동 설정
  - 데이터셋 전처리 자동화

- **makegraph.py**: 학습 메트릭 시각화 도구
  - CPU/GPU 사용량 그래프 생성
  - 학습 시간 및 메모리 사용량 분석
  - results/ 폴더의 JSON 파일을 기반으로 시각화
  - graphs/ 폴더에 PNG 형식으로 그래프 저장

- **label_csv_sender.py**: 공격자 패킷 구현 도구
  - CICIDS2017 데이터셋을 기반으로 공격자 패킷 생성
  - 선택된 공격 유형의 패킷을 클라이언트로 전송
  - TCP 소켓을 통한 실시간 패킷 전송

## 요구사항

- **Python 환경**
  - Python 3.7 이상 설치
  - pip3 패키지 관리자 설치
  - 가상환경 사용 권장 (venv 또는 conda)

- **Docker 환경**
  - Docker Engine 설치
  - Docker Compose 설치
  - Docker 네트워크 설정 권한

- **패키지 의존성**
  - 각 폴더의 `requirements.txt` 파일 참조
  - 주요 패키지:
    - pandas, numpy: 데이터 처리
    - scikit-learn: 머신러닝
    - matplotlib, seaborn: 시각화
    - python-dotenv: 환경 변수 관리
