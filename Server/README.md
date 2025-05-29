# Server

이 폴더는 연합학습의 서버 측 구현을 포함하고 있습니다.

## 폴더 구조

```
Server/
├── fl_server.py          # 연합학습 서버 구현
├── requirements.txt      # 필요한 Python 패키지 목록
└── Dockerfile           # 도커 이미지 빌드 설정
```

## 주요 파일 설명

### 서버 구현
- `fl_server.py`: 연합학습 서버의 메인 구현
  - Mobius IoT 플랫폼 통신
    - AE(AE) 감지 및 구독 (aeWatcher)
    - 컨테이너(CNT) 구독 및 관리
    - 클라이언트와의 pub/sub 통신
  - 클라이언트와의 통신 처리
    - 클라이언트로부터 모델 가중치 수신 (subscribe)
    - 글로벌 모델 전송 (publish)
    - 학습 결과 및 메트릭 수집
  - 연합학습 알고리즘
    - FedAvg 알고리즘 구현
    - 클라이언트 모델 가중치 집계
    - 글로벌 모델 업데이트
  - 모델 관리
    - 글로벌 모델 초기화 및 저장
    - 모델 성능 평가
    - 학습 진행 상황 모니터링
  - 자동화된 기능
    - 학습 결과 자동 저장
    - 오류 복구 및 재연결
    - 클라이언트 상태 모니터링

### 도커 관련
- `Dockerfile`: 서버 도커 이미지 빌드 설정
  - Python 3.7 기반 이미지
  - 필요한 패키지 설치
  - 실행 환경 구성

## 의존성
필요한 패키지는 `requirements.txt` 파일에 명시되어 있습니다. 주요 패키지:
- TensorFlow: 딥러닝 프레임워크
- NumPy: 수치 연산
- Pandas: 데이터 처리
- scikit-learn: 머신러닝 유틸리티
- psutil: 시스템 리소스 모니터링
- Flask: 웹 서버
- Flask-SocketIO: 실시간 통신
- requests: HTTP 클라이언트 