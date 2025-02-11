#!/bin/bash
# 이 스크립트는 .env 파일 혹은 인자로부터 CLIENT와 DOCKER_NETWORK 변수를 읽어,
# docker-compose.yaml 파일을 생성한 후 template.docker 파일의 내용을 반복해서 추가하고,
# 또한 각 Client 폴더의 Dockerfile 내의 __CLIENT_NUMBER__ 플레이스홀더를 치환합니다.

# .env 파일이 존재하면 환경변수 로드 (.env 파일에는 CLIENT와 DOCKER_NETWORK 값이 포함되어야 함)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# 불필요한 공백, 개행(\n,\r) 제거
CLIENT=$(echo "$CLIENT" | tr -d '\n\r ')
DOCKER_NETWORK=$(echo "$DOCKER_NETWORK" | tr -d '\n\r ')

# .env에 CLIENT 값이 없다면, 스크립트의 첫 번째 인자를 사용
if [ -z "$CLIENT" ]; then
  if [ -z "$1" ]; then
    echo "Usage: $0 CLIENT_COUNT"
    exit 1
  fi
  CLIENT="$1"
fi

# DOCKER_NETWORK 값이 없으면 기본값 지정 (마지막에 '.'이 있어야 함)
if [ -z "$DOCKER_NETWORK" ]; then
  DOCKER_NETWORK="172.20.0."
fi

echo "생성할 Client 컨테이너 수: $CLIENT"
echo "Docker Network IP prefix: $DOCKER_NETWORK"

COMPOSE_FILE="docker-compose.yaml"
TEMPLATE_FILE="template/cl_template.docker"
BASE_FILE="template/base_compose.docker"

# docker-compose.yaml 파일이 존재한다면, 이미 설정된 것으로 판단
if [ -f "$COMPOSE_FILE" ]; then
  echo "docker-compose.yaml 파일이 이미 존재합니다. 설정이 이미 완료된 상태입니다."
  exit 0
fi

# 기본 네트워크와 서비스(서버, ncube)의 IP 계산
GATEWAY="${DOCKER_NETWORK}1"
SERVER_IP="${DOCKER_NETWORK}2"
NCUBE_IP="${DOCKER_NETWORK}3"

# 기본 docker-compose 설정 파일(base_compose.yaml)이 존재하는지 확인 후,
# sed를 통해 변수 치환하여 docker-compose.yaml 파일 생성
if [ ! -f "$BASE_FILE" ]; then
  echo "Error: 기본 compose 파일($BASE_FILE)이 존재하지 않습니다."
  exit 1
fi

sed -e "s/\${GATEWAY}/$GATEWAY/g" \
    -e "s/\${SERVER_IP}/$SERVER_IP/g" \
    -e "s/\${NCUBE_IP}/$NCUBE_IP/g" \
    -e "s/\${DOCKER_NETWORK}/$DOCKER_NETWORK/g" "$BASE_FILE" > "$COMPOSE_FILE"

# docker-compose.yaml 파일 내의 services: 블록에 Client 항목을 추가
# 템플릿 파일(template.docker)에 있는 {NUM}과 {IP} 플레이스홀더를 치환하여 추가합니다.
for (( i = 1; i <= CLIENT; i++ )); do
  # Client IP 계산: 기본 IP 접두사 + (100 + i)
  IP_SUFFIX=$((100 + i))
  CLIENT_IP="${DOCKER_NETWORK}${IP_SUFFIX}"

  echo "Client${i} 설정 (IP: ${CLIENT_IP}) 추가 중..."
  sed -e "s/{NUM}/$i/g" -e "s/{IP}/$CLIENT_IP/g" "$TEMPLATE_FILE" >> "$COMPOSE_FILE"
done

# 각 Client 폴더 내의 Dockerfile 수정 (플레이스홀더 __CLIENT_NUMBER__를 치환)
for (( i = 1; i <= CLIENT; i++ )); do
  CLIENT_DOCKERFILE="Client${i}/Dockerfile"
  if [ -f "$CLIENT_DOCKERFILE" ]; then
    sed -i "s/__CLIENT_NUMBER__/$i/g" "$CLIENT_DOCKERFILE"
    echo "Modified $CLIENT_DOCKERFILE: __CLIENT_NUMBER__ -> $i"
  else
    echo "$CLIENT_DOCKERFILE 파일이 존재하지 않습니다."
  fi
done

# 현재 스크립트 위치 기준으로 Client 폴더로 이동
cd "$(dirname "$0")/Client" || { echo "Client 폴더로 이동 실패"; exit 1; }

# Client 폴더 내부에서 DatasetPreprocessCICIDS2017.py 스크립트 실행
python DatasetPreprocessCICIDS2017.py "$CLIENT"

echo "설정 완료: docker-compose.yaml 파일이 생성되었습니다."