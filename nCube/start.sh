#!/bin/sh

# 각각의 프로그램을 백그라운드 실행
node thyme.js &
node tas_emulator_FL2.js &

# 모든 백그라운드 프로세스가 종료될 때까지 대기
wait 