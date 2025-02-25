# Stage 1: eclipse-mosquitto 이미지에서 Mosquitto 관련 파일 및 라이브러리 복사
FROM eclipse-mosquitto:1.6 AS mosq

# Stage 2: node slim 이미지를 최종 베이스로 사용
FROM node:20.18.0-slim

# eclipse-mosquitto에서 Mosquitto 바이너리와 설정 파일 복사
COPY --from=mosq /usr/sbin/mosquitto /usr/sbin/mosquitto
COPY --from=mosq /mosquitto/config /etc/mosquitto

# Mosquitto 바이너리가 요구하는 musl 동적 링커와 libc를 /lib에 복사
COPY --from=mosq /lib/ld-musl-x86_64.so.1 /lib/ld-musl-x86_64.so.1
COPY --from=mosq /lib/libc.musl-x86_64.so.1 /lib/libc.musl-x86_64.so.1

# Mosquitto 바이너리가 요구하는 libssl과 libcrypto를 /opt/musl-libs에 복사
RUN mkdir -p /opt/musl-libs
COPY --from=mosq /usr/lib/libssl.so.48 /opt/musl-libs/
COPY --from=mosq /usr/lib/libcrypto.so.46 /opt/musl-libs/

# 필요한 패키지(예: supervisor, curl) 설치 후 mosquitto 사용자 생성
RUN apt-get update && \
    apt-get install -y supervisor curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -r mosquitto

# 작업 디렉토리 설정 (Mobius 애플리케이션 위치)
WORKDIR /home/node/app

# Node 의존성 설치를 위한 package 파일 복사 및 설치
COPY package*.json ./
RUN npm install

# Mobius 애플리케이션 전체 소스 복사
COPY . .

# Supervisor와 Mosquitto 로그용, 애플리케이션 로그 디렉토리 생성
RUN mkdir -p /home/node/app/log /var/log/mosquitto

# Supervisor 설정 파일 복사  
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 포트 노출: 7579 (Mobius 애플리케이션), 1883 (Mosquitto 브로커)
EXPOSE 7579 1883

# Supervisor를 PID1로 실행하여 Mosquitto와 Node 프로세스를 동시에 관리
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]