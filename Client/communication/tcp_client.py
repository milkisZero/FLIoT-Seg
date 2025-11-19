"""
TCP 클라이언트 (원본 메소드 유지)
"""
import socket
import struct
import json
import pickle
import time

class TCPClient:    
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.message_handle = None
        
    def connect(self):
        try:
            self.tcp_socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"TCP 소켓 연결 성공: {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print("TCP 소켓 연결 실패:", e)
            return False
    
    def disconnect(self):
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
                self.connected = False
            except:
                pass
    
    def send_tcp_message(self, header, message):
        try:
            if not isinstance(header, bytes):
                header=header.encode()
            if not isinstance(message, bytes):
                message_encode = message.encode()
            self.tcp_socket.send(header)
            self.tcp_socket.send(struct.pack('>I', len(message_encode)))
            self.tcp_socket.sendall(message_encode)
        except Exception as e:
            print(f"Error sending message: {e}")
    
    def send_blob_data(self, data, filename):
        header = b'WEIGHTS'
        blob_data = pickle.dumps(data)
        self.tcp_socket.send(header)
        self.tcp_socket.send(struct.pack('>I', len(blob_data)))
        self.tcp_socket.sendall(blob_data)
    
    def send_additional_metrics(self, metrics):
        try:
            header = b'METRICS'
            message = json.dumps(metrics)
            self.tcp_socket.send(header)
            self.tcp_socket.sendall(struct.pack('>I', len(message)) + message.encode())
            print(f"Additional metrics sent to {self.server_host}:{self.server_port}")
        except Exception as e:
            print(f"Error sending additional metrics: {e}")
    
    def send_additional_results(self, results):
        try:
            header = b'RESULTS'
            message = json.dumps(results)
            self.tcp_socket.send(header)
            self.tcp_socket.send(struct.pack('>I', len(message)) + message.encode())
            print(f"Additional results sent to {self.server_host}:{self.server_port}: {results}")
        except Exception as e:
            print(f"Error sending additional results: {e}")
      
    def receive_tcp_messages(self):
        def recv_exactly(size):
            # 정확히 size 바이트만큼 수신 
            data = b""
            while len(data) < size:
                chunk = self.tcp_socket.recv(size - len(data))
                if not chunk:
                    raise ConnectionError("연결이 끊어졌습니다.")
                data += chunk
                # print('chunk: ' , len(chunk))
                # print('total: ' , len(data))
            return data
         
        while True:
            try:
                message_length_bytes = recv_exactly(4)
                message_length = int.from_bytes(message_length_bytes, byteorder='big')
                print('message_length sum : ', message_length)
                #json_message = self.tcp_socket.recv(65535).decode('utf-8')

                json_message = recv_exactly(message_length)                 
                message_data = json.loads(json_message)

                try:
                    header = message_data['header']
                    message = message_data['message']
                    if hasattr(self, 'message_handler'):
                        self.message_handler(header, message)
                except Exception as e:
                    print(f"Handler error for header={header}: {e}")
                    break
            
            except Exception as e:
                print("Error receiving message:", e)
                break