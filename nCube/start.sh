#!/bin/sh

# PM2를 사용해 각각의 노드 프로그램 실행 (프로세스 이름을 지정하여 관리)
pm2 start thyme.js --name thyme
pm2 start tas_emulator_FL2.js --name tas_emulator

# 모든 PM2 프로세스의 로그를 실시간으로 출력
pm2 logs 