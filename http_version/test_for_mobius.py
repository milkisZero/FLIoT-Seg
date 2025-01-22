from flask import Flask, request, jsonify
import threading
import requests
import time

# Flask 앱 생성
app = Flask(__name__)

MOBIUS_URL = "http://192.168.0.60:7579"
HEADERS = {
    "X-M2M-Origin": "SOrigin",
    "X-M2M-RI": "12345",
    "Content-Type": "application/json"
}

SUB_CNT_List = ['client1FromC', 'client2FromC']

# 컨테이너 이름 , 이름/set == 토픽
PUB_CNT_List = ['client1FromS', 'client2FromS']

# Mobius를 구독해서 ae 감지
def create_aeWatcher():
    # 구독 요청 데이터
    payload = {
        "m2m:sub": {
            "rn": "aeWatcher",  # 구독 이름
            "nu": ["http://192.168.0.200:5000/aeWatcher"],  # 알림을 받을 서버 URL
            "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
            "enc": {
                "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
            },
        }
    }
    
    url = MOBIUS_URL + '/Mobius'
    headers = HEADERS
    headers["Content-Type"] = "application/json; ty=23"

    try:
        # POST 요청으로 구독 생성
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 201:
            print("aeWatcher created successfully!")
        else:
            print(f"Failed to create aeWatcher: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error creating aeWatcher: {e}")

# ae 구독
def create_aeSub(path):
    # Mobius 서버 및 리소스 경로 설정
  
    # 구독 요청 데이터
    payload = {
        "m2m:sub": {
            "rn": "aeSub",  # 구독 이름
            "nu": ["http://192.168.0.200:5000/aeSub"],  # 알림을 받을 서버 URL
            "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
            "enc": {
                "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
            },
            "exc": 100  # 최대 알림 횟수
        }
    }

    mobius_url = MOBIUS_URL + '/Mobius/' + path 
    headers = HEADERS
    headers["Content-Type"] = "application/json; ty=23"
    
    try:
        # POST 요청으로 구독 생성
        response = requests.post(mobius_url, headers=HEADERS, json=payload)

        if response.status_code == 201:
            print("aeSub created successfully!: ", path)
        else:
            print(f"Failed to create aeSub: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error creating aeSub: {e}")

# 컨테이너 구독
def create_cntSub(path):
    # Mobius 서버 및 리소스 경로 설정
  
    # 구독 요청 데이터
    payload = {
        "m2m:sub": {
            "rn": "cntSub",  # 구독 이름
            "nu": ["http://192.168.0.200:5000/notify"],  # 알림을 받을 서버 URL
            "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
            "enc": {
                "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
            },
            "exc": 100  # 최대 알림 횟수
        }
    }

    mobius_url = MOBIUS_URL + '/' + path 
    headers = HEADERS
    headers["Content-Type"] = "application/json; ty=23"
    
    try:
        # POST 요청으로 구독 생성
        response = requests.post(mobius_url, headers=HEADERS, json=payload)

        if response.status_code == 201:
            print("cntSub created successfully!: ", path)
        else:
            print(f"Failed to create cntSub: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error creating cntSub: {e}")

def create_cntPub():
    # pub 컨테이너 이름 == conf의 sub, 키 이름]
    for e in PUB_CNT_List:
        payload = {
            "m2m:cnt": {
                "rn": e,   
            }
        }
                
        url = MOBIUS_URL + '/Mobius/FLIoT'
        headers = HEADERS
        headers["Content-Type"] = "application/json; ty=3"

        try:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 201:
                print("cntPub created successfully!")
            else:
                print(f"Failed to create cntPub: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Error creating cntPub: {e}")
            return
    
    
        payload = {
            "m2m:sub": {
                "rn": "FLserverSub",  # 구독 이름
                "nu": ["mqtt://192.168.0.60:1883/SFLIoT/{e}/set?ct=json"],  # 알림을 받을 서버 URL
                "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
                "enc": {
                    "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
                },
                "exc": 100  # 최대 알림 횟수
            }
        }
        
        url = MOBIUS_URL + '/Mobius/FLIoT/' + e
        headers = HEADERS
        headers["Content-Type"] = "application/json; ty=23"

        try:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 201:
                print("sub created successfully!")
            else:
                print(f"Failed to create sub: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Error creating sub: {e}")

# aeWatcher 알림 엔드포인트
@app.route('/aeWatcher', methods=['POST'])
def aeWatcher():
    try:
        data = request.json
        print("Received AE:")

        if "m2m:sgn" in data:
            # rn인지 ri인지 선택해야 함
            content = data["m2m:sgn"]["nev"]["rep"]["m2m:ae"]["rn"]
            print(f"New AE detected(rn): {content}")
            create_aeSub(content)
            create_cntPub()
            
        return jsonify({"status": "received"}), 200

    except Exception as e:
        print(f"Error processing notification: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# aeSub 알림 엔드포인트
@app.route('/aeSub', methods=['POST'])
def aeSub():
    try:
        data = request.json
        print("Received Container:")

        if "m2m:sgn" in data:
            # rn인지 ri인지 선택해야 함
            content = data["m2m:sgn"]["nev"]["rep"]["m2m:cnt"]["rn"]
            print(f"New CNT detected(rn): {content}")
            url = data["m2m:sgn"]['sur']
            url = url[:url.rfind('/')+1]
            
            if content in SUB_CNT_List:
                create_cntSub(url + content)
            
        return jsonify({"status": "received"}), 200

    except Exception as e:
        print(f"Error processing notification: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 데이터 엔드포인트
@app.route('/notify', methods=['POST'])
def notify():
    try:
        # Mobius에서 전송된 알림 데이터
        data = request.json
        print("Received Notification:")

        # 알림 데이터에서 콘텐츠 추출
        if "m2m:sgn" in data:
            content = data["m2m:sgn"]["nev"]["rep"]["m2m:cin"]["con"]
            print("Updated Content: ", content)   

        return jsonify({"status": "received"}), 200

    except Exception as e:
        print(f"Error processing notification: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
# Flask 서버 실행
def run_flask():
    app.run(host="192.168.0.200", port=5000)

if __name__ == "__main__":
    # Flask 서버를 별도의 스레드에서 실행
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Mobius 구독 생성
    time.sleep(1)  # Flask 서버가 준비될 때까지 대기
    create_aeWatcher()
    create_cntPub()

    # 메인 스레드는 Flask 서버가 실행되는 동안 대기
    while True:
        time.sleep(1)

