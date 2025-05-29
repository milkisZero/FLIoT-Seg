# Client

이 폴더는 연합학습의 클라이언트 측 구현을 포함하고 있습니다.

## 폴더 구조

```
Client/
├── 💚 CICIDS/            # CICIDS2017 데이터셋 저장소 (사용자 추가 필요)
├── 💙 CICIDS_Splitted/   # 전처리된 데이터셋 저장소 (자동 생성)
├── datasetCICIDS.py      # 데이터셋 전처리 유틸리티
├── DatasetPreprocessCICIDS2017.py  # 데이터셋 전처리 스크립트
├── fl_client.py          # 연합학습 클라이언트 구현
├── pickle_utils.py       # Pickle 데이터 변환 유틸리티
├── requirements.txt      # 필요한 Python 패키지 목록
├── Dockerfile           # 도커 이미지 빌드 설정
└── .dockerignore        # 도커 빌드 제외 파일 목록
```

> 💚 사용자가 직접 추가해야 하는 폴더/파일
> 💙 자동으로 생성되는 폴더/파일

## 주요 파일 설명

### 데이터셋 관련
- `datasetCICIDS.py`: CICIDS2017 데이터셋을 처리하기 위한 유틸리티 클래스와 함수들을 포함
  - 데이터 로딩, 전처리, 정규화 등의 기능 제공
  - 연합학습에 적합한 형태로 데이터 변환
  - 데이터 증강 및 샘플링 기능

- `DatasetPreprocessCICIDS2017.py`: CICIDS2017 데이터셋 전처리 스크립트
  - 원본 CSV 파일을 클라이언트별로 분할
  - 각 클라이언트의 학습/테스트 데이터셋 생성
  - 데이터 정규화 및 특성 선택
  - 전처리된 데이터는 `CICIDS_Splitted/` 폴더에 저장

  사용 방법:
  ```bash
  # 1. CICIDS2017 데이터셋을 CICIDS/ 폴더에 저장
  
  # 2. 환경 변수 설정
  # .env.example 파일을 참조하여 필요한 환경 변수를 설정합니다.
  # 주요 설정 항목:
  # - LABEL: 선택할 공격 유형 인덱스
  # - BENIGN_PACKETS: 정상 패킷 수
  # - ATTACK_PACKETS: 공격 패킷 수
  # - RATIO: 클라이언트별 데이터 비율
  # - TR_RATIO: 클라이언트별 학습 데이터 비율
  # - SHUFFLE_INTENSITY: 데이터 셔플 강도
  
  # 3. 스크립트 실행 (Client 폴더에서)
  cd Client
  source ../.env
  python3 DatasetPreprocessCICIDS2017.py $CLIENT
  ```
  
  출력 파일:
  - `CICIDS_Splitted/client{1..N}_train.csv`: 각 클라이언트의 학습 데이터
  - `CICIDS_Splitted/client{1..N}_test.csv`: 각 클라이언트의 테스트 데이터
  - `CICIDS_Splitted/splitting_info.txt`: 데이터 분할 정보
  - `CICIDS_Splitted/all_clients_distribution.png`: 클라이언트별 데이터 분포 그래프
  - `Server/config.json`: 선택된 라벨 정보

### 클라이언트 구현
- `fl_client.py`: 연합학습 클라이언트의 메인 구현
  - 서버와의 통신 처리
    - Gateway 서버와 TCP 소켓 통신
    - 모델 가중치 및 학습 결과 전송
    - 실시간 학습 메트릭 전송 (loss, accuracy 등)
    - Gateway를 통한 중앙 서버와의 통신
  - 공격자 패킷 처리
    - 공격자 클라이언트의 TCP 연결 수신 (포트 4000)
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

### 유틸리티
- `pickle_utils.py`: Pickle 데이터 변환 유틸리티
  - 모델 파라미터의 직렬화/역직렬화
  - 네트워크 전송을 위한 데이터 변환
  - 메모리 효율적인 데이터 처리

### 도커 관련
- `Dockerfile`: 클라이언트 도커 이미지 빌드 설정
  - Python 3.7 기반 이미지
  - 필요한 패키지 설치
  - 실행 환경 구성

- `.dockerignore`: 도커 빌드 시 제외할 파일 목록
  - 불필요한 파일 제외로 이미지 크기 최적화

## 폴더 구조
- 💚 `CICIDS/`: 원본 CICIDS2017 데이터셋 저장 (사용자 추가 필요)
  - CSV 형식의 원본 데이터 파일들
- 💙 `CICIDS_Splitted/`: 전처리된 데이터셋 저장 (자동 생성)
  - 각 클라이언트별 학습/테스트 데이터 포함
  - 정규화된 특성과 레이블

## 의존성
필요한 패키지는 `requirements.txt` 파일에 명시되어 있습니다. 주요 패키지:
- TensorFlow: 딥러닝 프레임워크
- NumPy: 수치 연산
- Pandas: 데이터 처리
- scikit-learn: 머신러닝 유틸리티
- psutil: 시스템 리소스 모니터링 