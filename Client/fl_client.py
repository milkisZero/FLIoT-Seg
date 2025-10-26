"""
Federated Learning Client (원본 메소드 재사용)
"""
import os
import sys
import json
import time
import datetime
import threading
from typing import Optional
from communication.tcp_client import TCPClient
from federated.protocol_handler import FLProtocolHandler
from federated.result_manager import FLResultManager
from utils.gpu_setup import setup_gpu

class FederatedClient:
    """원본 FederatedClient를 TCPClient + FLProtocolHandler로 분리하여 재구성"""
        
    def __init__(self, server_host, server_port, time_start, benign_train_only=False):
        """원본 __init__ 로직 유지"""
        self.benign_train_only = benign_train_only
        self.local_model = None
        self.stop_training = False
        self.file_end = False
        self.time_start = time_start
        
        setup_gpu(gpu_id=0, enable_mixed_precision=True)
        
        # TCP 클라이언트 생성 (원본 로직 분리)
        self.tcp_client = TCPClient(server_host, server_port)
        
        if not self.tcp_client.connect():
            print("TCP 연결 실패")
            return
        
        # 스레드 락
        self.socket_lock = threading.Lock()
        
        # 실행 폴더 생성 (원본 라인 19-35 그대로)
        self.execution_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.execution_folder):
            os.makedirs(self.execution_folder)
        
        # JSON 파일 생성 (원본 그대로)
        import tensorflow as tf
        device = "gpu" if tf.config.list_physical_devices("GPU") and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        self.json_file_name = os.path.join("results", device, f"{self.execution_folder}.json")
        if not os.path.exists(os.path.dirname(self.json_file_name)):
            os.makedirs(os.path.dirname(self.json_file_name))
        
        with open(self.json_file_name, "w") as f:
            json.dump([], f, indent=4)
        
        print(f"데이터가 저장될 폴더: {self.execution_folder}")
        print(f"JSON 파일이 생성되었습니다: {self.json_file_name}")
        
        self.result_manager = FLResultManager(base_dir= "../results", execution_folder = self.execution_folder)
        
        # 프로토콜 핸들러 생성
        self.protocol_handler = FLProtocolHandler(
            self.tcp_client, 
            self.time_start,
            self.result_manager,
            self.execution_folder,
        )
        
        # 원본 라인 43-48 그대로
        print("sent wakeup")
        message = json.dumps({
            'event': 'client_wake_up'
        })
        self.tcp_client.send_tcp_message('OPERATE', message)
        
        # 메시지 수신 시작 (원본 그대로)
        self.tcp_client.receive_tcp_messages()
    
if __name__ == "__main__":        
    # TCP client settings
    if os.environ.get('CLIENT') is not None:
        TCP_SERVER_IP = 'gateway'  # IP address of the receiving Raspberry Pi
        TCP_SERVER_PORT = 3105
    else:
        TCP_SERVER_IP = '192.168.0.10'  # 수신 라즈베리파이의 IP 주소
        TCP_SERVER_PORT = 3105
    
    # 클라이언트 실행 (원본과 동일한 방식)
    time_start = time.time()
    client = FederatedClient(TCP_SERVER_IP, TCP_SERVER_PORT, time_start)