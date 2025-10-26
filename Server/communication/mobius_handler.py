import requests
from typing import List, Dict

# MOBIUS_URL = "http://mobius:7579"
# HEADERS = {
#     "X-M2M-Origin": "SOrigin",
#     "X-M2M-RI": "12345",
#     "Content-Type": "application/json"
# }

class MobiusHandler:    
    def __init__(self, config):
        self.MOBIUS_URL = config.mobius_url
        self.HEADERS = config.mobius_headers
        self.host = config.host
        self.port = config.port
        
        # 구독 목록
        self.sub_cnt_list: List[str] = []
        self.pub_cnt_list: List[str] = []
    
        # Mobius를 구독해서 ae 감지
    def create_aeWatcher(self):
        # 구독 요청 데이터
        payload = {
            "m2m:sub": {
                "rn": "aeWatcher",  # 구독 이름
                "nu": [f"http://{self.host}:{self.port}/aeWatcher"],  # 알림을 받을 서버 URL
                "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
                "enc": {
                    "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
                },
            }
        }
        
        url = self.MOBIUS_URL + '/Mobius'
        headers = self.HEADERS
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
    def create_aeSub(self,path):
        # Mobius 서버 및 리소스 경로 설정
    
        # 구독 요청 데이터
        payload = {
            "m2m:sub": {
                "rn": "aeSub",  # 구독 이름
                "nu": [f"http://{self.host}:{self.port}/aeSub"],  # 알림을 받을 서버 URL
                "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
                "enc": {
                    "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
                },
                "exc": 100  # 최대 알림 횟수
            }
        }

        mobius_url = self.MOBIUS_URL + '/Mobius/' + path 
        headers = self.HEADERS
        headers["Content-Type"] = "application/json; ty=23"
        
        try:
            # POST 요청으로 구독 생성
            response = requests.post(mobius_url, headers=headers, json=payload)

            if response.status_code == 201:
                print("aeSub created successfully!: ", path)
            else:
                print(f"Failed to create aeSub: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Error creating aeSub: {e}")

    # 컨테이너 구독
    def create_cntSub(self,path,content):
        self.sub_cnt_list.append(content)
        # Mobius 서버 및 리소스 경로 설정
    
        # 구독 요청 데이터
        payload = {
            "m2m:sub": {
                "rn": "cntSub",  # 구독 이름
                "nu": [f"http://{self.host}:{self.port}/notify"],  # 알림을 받을 서버 URL
                "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
                "enc": {
                    "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
                },
                "exc": 100  # 최대 알림 횟수
            }
        }

        mobius_url = self.MOBIUS_URL + '/' + path 
        headers = self.HEADERS
        headers["Content-Type"] = "application/json; ty=23"
        
        try:
            # POST 요청으로 구독 생성
            response = requests.post(mobius_url, headers=headers, json=payload)

            if response.status_code == 201:
                print("cntSub created successfully!: ", path)
            else:
                print(f"Failed to create cntSub: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Error creating cntSub: {e}")

    def publish(self, path, data):
        payload = {
            "m2m:cin": {
                "con": data,   
            }
        }
        print("publish path: ", path)
        url = self.MOBIUS_URL + '/Mobius/FLIoT/' + path
        headers = self.HEADERS
        headers["Content-Type"] = "application/json; ty=4"

        try:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 201:
                print("Publish successfully!", url)
            else:
                print(f"Failed to Publish: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Error publishing: {e}")
    