# Linux 환경에서 PyInstaller 설정

## 1. 필수 패키지 설치
```bash
sudo apt update
sudo apt install software-properties-common
```

## 2. Python 3.7 설치
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7
```

## 3. Python 3.7 기본 설정
```bash
sudo update-alternatives --config python3
```

## 4. pip 설치
```bash
curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py
sudo python3.7 get-pip.py
```

## 5. PyInstaller 및 요구 사항 설치
```bash
cd Client
pip3.7 install -r requirements.txt
pip3.7 install pyinstaller
```

## 6. PyInstaller 실행
`pyinstaller.command` 파일을 사용하여 실행