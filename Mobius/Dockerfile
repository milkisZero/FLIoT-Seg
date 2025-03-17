# Stage 1: eclipse-mosquitto 이미지에서 Mosquitto 관련 파일 및 라이브러리 복사
FROM eclipse-mosquitto:1.6 AS mosq

# Stage 2: node slim 이미지를 최종 베이스로 사용
FROM node:20.18.0-slim

# eclipse-mosquitto에서 Mosquitto 바이너리와 설정 파일 복사
COPY --from=mosq /usr/sbin/mosquitto /usr/sbin/mosquitto
COPY --from=mosq /mosquitto/config /etc/mosquitto

# Mosquitto 바이너리가 요구하는 모든 필요한 라이브러리 복사
COPY --from=mosq /usr/lib/libssl.so* /usr/lib/
COPY --from=mosq /usr/lib/libcrypto.so* /usr/lib/
COPY --from=mosq /lib/ld-musl-*.so* /lib/

# OpenSSL 설정 파일 복사
COPY --from=mosq /etc/ssl /etc/ssl

# 단계 1: 시스템 업데이트 및 업그레이드 후 기본 패키지 설치
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends supervisor curl python3

# 단계 2: 나머지 패키지 설치 (make, g++, openssl) - 비대화형 모드 적용 및 dpkg 옵션 추가
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Options::="--force-confnew" install -y --no-install-recommends make g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# openssl 별도 설치 (설치 오류 발생 시 dpkg --configure -a 후 설치)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Options::="--force-confnew" install -y --no-install-recommends openssl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 사용자 생성 (Debian slim에서는 --system 옵션 사용)
RUN adduser --system --no-create-home --group mosquitto

# 작업 디렉토리 설정 (Mobius 애플리케이션 위치)
WORKDIR /home/node/app

# Node 의존성 설치를 위한 package 파일 복사 및 설치
COPY package*.json ./
RUN npm install

# Mobius 애플리케이션 전체 소스 복사
COPY . .

# Supervisor와 Mosquitto 로그용, 애플리케이션 로그 디렉토리 생성
RUN mkdir -p /home/node/app/log /var/log/mosquitto && \
    chown -R mosquitto:root /var/log/mosquitto

# Mosquitto 설정 수정 (필요한 경우)
RUN echo "listener 1883 0.0.0.0" >> /etc/mosquitto/mosquitto.conf && \
    echo "allow_anonymous true" >> /etc/mosquitto/mosquitto.conf

# supervisord.conf 파일 수정하여 Mosquitto 환경 변수 추가
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf 
RUN grep -q "environment=" /etc/supervisor/conf.d/supervisord.conf || \
    sed -i '/\[program:mosquitto\]/a environment=LD_LIBRARY_PATH=/usr/lib:/lib' /etc/supervisor/conf.d/supervisord.conf

# Supervisor를 PID1로 실행하여 Mosquitto와 Node 프로세스를 동시에 관리
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]