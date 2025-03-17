import warnings
warnings.filterwarnings("ignore")

import os
import tensorflow as tf
import psutil
import subprocess

# Check available GPU list
print("CLIENT: ", os.environ.get('CLIENT'))
gpus = tf.config.list_physical_devices('GPU')
if os.environ.get('CLIENT') is not None or not gpus:
    print("No available GPU. Use CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    print("Available GPU list:")
    for idx, gpu in enumerate(gpus):
        print(f"  {idx}: {gpu.name}")
        
    selected_gpu = input("Select the index of the GPU to use (-1: use CPU): ").strip()
    try:
        selected_gpu = int(selected_gpu)
    except ValueError:
        print("Invalid input. Using default GPU 0.")
        selected_gpu = 0

    if selected_gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = INFO 필터, 2 = WARNING 필터, 3 = ERROR 필터

import numpy as np
import keras
import random
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from keras.utils.data_utils import get_file
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import csv
import time
import json
import pickle
import codecs
from keras.models import model_from_json
from pickle_utils import obj_to_pickle_string, pickle_string_to_obj
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,roc_curve,auc
from datasetCICIDS import gen_train_valid_data
import datetime,time
import socket
import struct
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import resource  # 메모리 측정을 위한 리소스 모듈 임포트

print("now is {}".format(datetime.datetime.today()))
datasource = gen_train_valid_data(benign_only=True)
# datasource가 6개의 항목으로 구성됨: X_train, y_train, X_test, y_test, orig_y_train, orig_y_test
data_to_save = {
    "X_train": datasource[0].tolist(),
    "y_train": datasource[1].tolist(),
    "X_test": datasource[2].tolist(),
    "y_test": datasource[3].tolist(),
    "original_y_train": datasource[4].tolist(),
    "original_y_test": datasource[5].tolist()
}

with open("datasource.json", "w") as f:
    json.dump(data_to_save, f, indent=4)
import threading
import signal
import sys

# TCP client settings
if os.environ.get('CLIENT') is not None:
    TCP_SERVER_IP = 'gateway'  # IP address of the receiving Raspberry Pi
    TCP_SERVER_PORT = 3105
else:
    TCP_SERVER_IP = '192.168.0.10'  # 수신 라즈베리파이의 IP 주소
    TCP_SERVER_PORT = 3105

class LocalModel(object):
    def __init__(self, model_config, data_collected):
        self.model_config = model_config
        self.model = model_from_json(model_config['model_json'])
        self.x_train, self.y_train, self.x_test, self.y_test, self.original_y_train, self.original_y_test = data_collected

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):
        start_time = time.time()
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=RMSprop(),
                           metrics=['accuracy'])
        self.loss = self.model.fit(
            self.x_train, self.x_train,
            epochs=self.model_config['epoch_per_round'],
            batch_size=self.model_config['batch_size'],
            validation_data=(self.x_test, self.x_test),
            verbose=2
        )
        end_time = time.time()
        train_time = end_time - start_time

        # 리소스를 이용하여 최고 메모리 사용량(킬로바이트 단위)을 측정하고 MB 단위로 변환
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_memory_mb = usage.ru_maxrss / 1024

        current_loss = self.loss.history['loss'][0]
        print('One round training loss: {:.16f}'.format(current_loss))
        print('이 라운드 학습 시간: {:.4f} 초, 최고 메모리 사용량: {:.2f} MB'.format(train_time, peak_memory_mb))
            
        return self.model.get_weights(), current_loss, train_time, peak_memory_mb

    def evaluate1(self):
        def calculate_losses(x, preds):
            losses = np.zeros(len(x))
            for i in range(len(x)):
                losses[i] = np.mean(np.square(preds[i] - x[i]))
            return losses

        # Always compute threshold using the current training set predictions.
        print("Computing threshold from training data predictions...")
        train_preds = self.model.predict(self.x_train, batch_size=16, verbose=1)
        train_losses = calculate_losses(self.x_train, train_preds)
        # Changed threshold percentile from 90 to 99 for better anomaly separation.
        threshold = np.percentile(train_losses, 99)
        print("Computed threshold:", threshold)

        testing_set_predictions = self.model.predict(self.x_test, verbose=1)
        test_losses = calculate_losses(self.x_test, testing_set_predictions)

        # Generate binary anomaly predictions.
        binary_predictions = np.zeros(len(test_losses))
        binary_predictions[test_losses > threshold] = 1

        precision = precision_score(self.y_test, binary_predictions)
        recall = recall_score(self.y_test, binary_predictions)
        f1 = f1_score(self.y_test, binary_predictions)
        print("Performance over the testing data set:")
        print("  Recall: {:.16f}, Precision: {:.16f}, F1: {:.16f}".format(recall, precision, f1))
        return f1, precision, recall

# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 1200
    def __init__(self, server_host, server_port, datasource, benign_train_only=True):
        self.benign_train_only = benign_train_only
        self.datasource = datasource
        self.local_model = None
        self.stop_training = False
        self.file_end = False
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.tcp_socket.connect((server_host, server_port))
        except Exception as e:
            print("TCP 소켓 연결 실패:", e)
            # 재연결 로직 추가
        
        self.eval_lock = threading.Lock()
        self.socket_lock = threading.Lock()
        
        # 프로그램 최초 실행 시간으로 폴더 생성 (예: 20231026_123456)
        self.execution_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.execution_folder):
            os.makedirs(self.execution_folder)
        print(f"데이터가 저장될 폴더: {self.execution_folder}")

        # 지속적인 테스트 평가를 위한 쓰레드 실행
        self.test_interval = 60
        self.testing_thread = threading.Thread(target=self.continuous_testing, daemon=True)
        self.testing_thread.start()

        # 공격자 클라이언트의 연결을 받을 소켓 설정 (예: 포트 4000 사용)
        self.setup_attacker_listener(attacker_port=4000)
        self.attacker_thread = threading.Thread(target=self.receive_attacker_messages, daemon=True)
        self.attacker_thread.start()

        # 기존 TCP 메시지 수신 쓰레드 시작
        print("sent wakeup")
        message = json.dumps({
            'event': 'client_wake_up'
        })
        self.send_tcp_message('OPERATE', message)
        self.receive_tcp_messages()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # 프로그램 초기에 한 번만 모든 GPU에 대해 메모리 증분 할당 활성화
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print("GPU 메모리 증분 할당 설정 중 오류 발생:", e)

    def setup_attacker_listener(self, attacker_port):
        """공격자 클라이언트의 연결을 수신하기 위한 서버 소켓을 설정합니다."""
        self.attacker_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.attacker_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.attacker_listener.bind(('', attacker_port))
        self.attacker_listener.listen(5)
        print(f"공격자 패킷 수신을 위한 포트({attacker_port})에서 대기 중입니다.")

    def receive_attacker_messages(self):
        """공격자(attacker) 클라이언트의 메시지를 수신 및 처리합니다.
        수신된 메시지는 'classify_packet' 이벤트로 처리되어,
        on_classify_packet 함수를 호출합니다.
        """
        while True:
            try:
                attacker_conn, attacker_addr = self.attacker_listener.accept()
                print("공격자 클라이언트 연결됨:", attacker_addr)
                while True:
                    # 먼저 4바이트 크기의 메시지 길이를 수신
                    message_length_bytes = self.recv_exactly_from(attacker_conn, 4)
                    message_length = int.from_bytes(message_length_bytes, byteorder='big')
                    print("공격자 메시지 예상 길이:", message_length)
                    
                    # 지정된 길이만큼의 데이터를 읽어들임
                    json_message = self.recv_exactly_from(attacker_conn, message_length)
                    message_data = json.loads(json_message)
                    header = message_data['header']
                    message = message_data['message']
                    
                    # 공격자 클라이언트의 메시지 처리 로직을 여기에 구현합니다.
                    self.handle_message(header, message)
            except Exception as e:
                if not self.file_end:
                    print("공격자 메시지 수신 중 오류:", e)
                    break

    def recv_exactly_from(self, conn, size):
        """특정 연결(conn)에서 정확히 size 바이트만큼 데이터 수신"""
        data = b""
        while len(data) < size:
            chunk = conn.recv(size - len(data))
            if not chunk:
                raise ConnectionError("연결이 끊어졌습니다.")
            data += chunk
        return data

    def continuous_testing(self):
        while not self.stop_training:
            time.sleep(self.test_interval)
            # Check that the local model exists and its evaluation method is callable
            if self.local_model is not None and hasattr(self.local_model, 'evaluate1') and callable(self.local_model.evaluate1):
                print("\n[Continuous Testing] Evaluating model on test data ...")
                try:
                    with self.eval_lock:
                        f1, precision, recall = self.local_model.evaluate1()
                    print("[Continuous Testing] Results -> F1: {:.6f}, Precision: {:.6f}, Recall: {:.6f}\n"
                          .format(f1, precision, recall))
                except Exception as e:
                    print("Error during continuous testing:", e)
            else:
                print("[Continuous Testing] Local model or evaluation method not ready.")

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
     #           print(message_data)
                header = message_data['header']
                message = message_data['message']

                self.handle_message(header, message)
            except Exception as e:
                print("Error receiving message:", e)
                break

    def handle_message(self, header, message):
        if header == 'OPERATE':
            event = message['event']
            if event == 'connect':
                self.on_connect()
            elif event == 'disconnect':
                self.on_disconnect()
            elif event == 'reconnect':
                self.on_reconnect()
            elif event == 'init':
                self.on_init(message['payload'])
            elif event == 'request_update':
                self.on_request_update(message['payload'])
            elif event == 'stop_and_eval' or event == 'request_eval':
                self.on_eval(message['payload'], event)
            elif event == 'global_update':
                self.on_global_update(message['payload'])
            elif event == 'classify_packet':
                self.on_classify_packet(message['payload'])
            elif event == 'file_end':
                self.on_file_end()
            else:
                print("Unknown event:", event)
        else:
            print("Unknown header:", header)

    def on_connect(self):
        print("Connected")

    def on_disconnect(self):
        print("Disconnected")

    def on_reconnect(self):
        print("Reconnected")

    def on_init(self, *args):
        model_config = args[0]
        print("Init message received:", model_config)
        
        # 모델 및 데이터셋 초기화
        self.local_model = LocalModel(model_config, self.datasource)
        
        if self.benign_train_only:
            # benign label(0) 데이터만 사용하도록 확인 및 추가 필터링
            indices = np.where(self.local_model.y_train == 0)[0]
            self.local_model.x_train = self.local_model.x_train[indices]
            self.local_model.y_train = self.local_model.y_train[indices]
            print("benign_train_only 플래그 활성화: benign 데이터만 학습에 사용합니다.")
        
        # model_json을 json 형식으로 변환하여 저장: 문자열을 딕셔너리로 변환
        try:
            model_config_converted = model_config.copy()
            model_config_converted["model_json"] = json.loads(model_config_converted["model_json"])
        except Exception as e:
            print("model_json 변환 오류:", e)
            model_config_converted = model_config
        
        # 모델 구성 정보(예, 에폭, 배치 사이즈, model_json 등)를 최초 실행 폴더에 저장 (한 번만 저장)
        config_file = os.path.join(self.execution_folder, "model_config.json")
        if not os.path.exists(config_file):
            with open(config_file, "w") as f:
                json.dump(model_config_converted, f, indent=4)
            print(f"모델 및 설정 정보가 {config_file} 에 저장되었습니다.")
        
        header = b'OPERATE'
        FL_ready = json.dumps({
            'event': 'client_ready',
            'payload': {
                'train_size': self.local_model.x_train.shape[0]
            }
        })
        self.send_tcp_message(header, FL_ready)

    def send_additional_results(self, results):
        try:
            header = b'RESULTS'
            message = json.dumps(results)
            self.tcp_socket.send(header)
            self.tcp_socket.send(struct.pack('>I', len(message)) + message.encode())
            print(f"Additional results sent to {TCP_SERVER_IP}:{TCP_SERVER_PORT}: {results}")
        except Exception as e:
            print(f"Error sending additional results: {e}")

    def on_global_update(self, *args):
        req = args[0]
        print("global update requested")

        self.local_model.model = model_from_json(req['model_json'])
        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])
        with self.eval_lock:
            self.local_model.set_weights(weights)

    def save_round_time(self, round_number, train_time, peak_memory):
        """
        각 라운드의 학습 시간, 최고 메모리 사용량, 라운드 번호, 그리고 실행 장치(CPU 또는 GPU)를
        프로그램 최초 실행 시간으로 생성된 폴더 내의 JSON 파일에 저장합니다.
        파일명은 사용 장치에 따라 "cpu_round_times.json" 또는 "gpu_round_times.json"으로 생성됩니다.
        """
        folder = self.execution_folder
        # 사용 장치 판별: CUDA_VISIBLE_DEVICES가 설정되어 있고, GPU 목록이 존재하면 GPU, 아니라면 CPU
        device = "gpu" if tf.config.list_physical_devices("GPU") and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        file_name = os.path.join(folder, f"{device}_round_times.json")

        # 기존 데이터 있으면 불러오기
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print("JSON 파일 로드 오류:", e)
                    data = []
        else:
            data = []

        # 새 데이터를 추가
        data.append({
            "round_number": round_number,
            "train_time": train_time,
            "peak_memory": peak_memory,
            "device": device
        })

        # 파일에 저장
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Round {round_number} 학습 시간 {train_time:.4f}초, 최고 메모리 사용량 {peak_memory:.2f} MB (장치: {device})가 {file_name} 에 저장되었습니다.")

    def on_request_update(self, *args):
        req = args[0]
        print("update requested")
        print('round_number:', req['round_number'])
    
        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])
    
        with self.eval_lock:
            self.local_model.set_weights(weights)
        
        # 학습 라운드 진행 시 시간 및 메모리 사용량 측정값 반환
        my_weights, train_loss, train_time, peak_memory = self.local_model.train_one_round()
    
        # 라운드 번호, 학습 시간, 최고 메모리 사용량, 장치 정보를 프로그램 최초 실행 폴더 내의 파일에 저장
        self.save_round_time(req['round_number'], train_time, peak_memory)
    
        # 전송하는 정보(payload)는 원래대로 전송합니다.
        header = b'OPERATE'
        resp = json.dumps({
            'event': 'client_update',
            'payload': {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'train_loss': train_loss
            }
        })
        self.send_tcp_message(header, resp)
    
        additional_metrics = {
            'round_number': req['round_number'],
            'loss': self.local_model.loss.history['loss'][-1],
            'accuracy': self.local_model.loss.history['accuracy'][-1],
            'val_loss': self.local_model.loss.history['val_loss'][-1],
            'val_accuracy': self.local_model.loss.history['val_accuracy'][-1]
        }
        self.send_additional_metrics(additional_metrics)

    def on_eval(self, *args):
        req = args[0]
        event = args[1]
        if(event == 'stop_and_eval'):
            self.stop_training = True
        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])
        with self.eval_lock:
            self.local_model.set_weights(weights)
        f1, precision, recall = self.local_model.evaluate1()
        time_end = time.time()
        print('\033[1;35;0m Time cost = %fs \033[0m' % (time_end - time_start))
        #test_loss, test_accuracy = self.local_model.evaluate()
        payload={
            'test_size': self.local_model.x_test.shape[0],
            #'test_loss': test_loss,
            #'test_accuracy': test_accuracy
        }
        if 'round_number' in req:
            payload['round_number'] = req['round_number']

        header = b'OPERATE'
        resp = json.dumps({
            'event': 'client_eval',
            'payload': payload
        })


        self.send_tcp_message(header, resp)

        additional_results = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        self.send_additional_results(additional_results)

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

#    def send_pickle_file(self, filename):
#        self.tcp_socket.send(filename.encode())

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
            print(f"Additional metrics sent to {TCP_SERVER_IP}:{TCP_SERVER_PORT}")
        except Exception as e:
            print(f"Error sending additional metrics: {e}")

    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))

    def on_classify_packet(self, payload):
        """
        이 메서드는 수신된 패킷 정보를 이용해 학습된 모델로 예측을 진행하고,
        예측한 라벨과 실제 라벨(true_label)이 일치하는지 판별합니다.
        
        payload 예시:
        {
            "packet": [0.1, 0.5, ...],   # 패킷 정보 (특징 데이터 리스트)
            "true_label": "nomaly" 또는 "anomaly"  # 실제 라벨
        }
        """
        if self.local_model is None:
            print("로컬 모델이 초기화되지 않았습니다. 먼저 모델을 초기화해주세요.")
            return
        try:
            # 수신된 데이터 확인
            packet = payload.get("packet")
            true_label = payload.get("true_label")
            
            # 패킷 데이터를 numpy 배열로 변환 (모델 입력 크기에 맞게 reshape)
            packet_array = np.array(packet)
            if packet_array.ndim == 1:
                packet_array = np.expand_dims(packet_array, axis=0)
            
            # 모델 예측(재구성) 수행
            prediction = self.local_model.model.predict(packet_array, verbose=0)
            
            # 재구성 오차 (Mean Squared Error) 계산
            error = np.mean(np.square(packet_array - prediction))
            
            # 임계값(threshold) 확인 - self.local_model.threshold 사용
            if not hasattr(self.local_model, 'threshold'):
                # threshold가 없는 경우 처음 한 번만 계산
                print("임계값 없음, 임계값 계산 중...")
                train_preds = self.local_model.model.predict(self.local_model.x_train, batch_size=16, verbose=0)
                train_losses = np.mean(np.square(self.local_model.x_train - train_preds), axis=1)
                self.local_model.threshold = np.percentile(train_losses, 99)
                print(f"임계값 계산 완료: {self.local_model.threshold:.6f}")
            
            threshold = self.local_model.threshold
            
            # nomaly/anomaly 분류
            result = "anomaly" if error > threshold else "nomaly"
            
            # 예측 결과와 실제 라벨 비교
            is_correct = result == true_label
            
            # 통계 초기화 (필요한 경우)
            if not hasattr(self, 'classification_stats'):
                self.classification_stats = {
                    'nomaly_correct_predictions': 0, 
                    'anomaly_correct_predictions': 0,
                    'nomaly_incorrect_predictions': 0,
                    'anomaly_incorrect_predictions': 0
                }
            
            # 통계 업데이트
            if is_correct:
                if true_label == 'nomaly':
                    self.classification_stats['nomaly_correct_predictions'] += 1
                else:
                    self.classification_stats['anomaly_correct_predictions'] += 1
            else:
                if true_label == 'nomaly':
                    self.classification_stats['nomaly_incorrect_predictions'] += 1
                else:
                    self.classification_stats['anomaly_incorrect_predictions'] += 1

            # 요청한 형식으로 결과 출력
            print(f"\n패킷 분류 결과:")
            print(f"  실제 라벨 (매핑 후): {true_label}")
            print(f"  예측 라벨 (nomaly/anomaly): {result}")
            print(f"  재구성 오차: {error:.6f}")
            print(f"  임계값 (99번째 백분위수): {threshold:.6f}")
            
            print(f"\n누적 예측 결과:")
            print(f"  nomaly 정답 개수: {self.classification_stats['nomaly_correct_predictions']}")
            print(f"  nomaly 오답 개수: {self.classification_stats['nomaly_incorrect_predictions']}")
            print(f"  anomaly 정답 개수: {self.classification_stats['anomaly_correct_predictions']}")
            print(f"  anomaly 오답 개수: {self.classification_stats['anomaly_incorrect_predictions']}")
            
            # 정답률 계산 (소수점 둘째 자리까지)
            accuracy = self.classification_stats['correct'] / self.classification_stats['total'] * 100
            print(f"  전체 정답률: {accuracy:.2f}%")
            
        except Exception as e:
            print("분류 중 오류 발생:", e)

    def on_file_end(self):
        try:
            header = b'ATTACKS'
            # 딕셔너리 키 이름 확인 (이전에 사용한 키 이름과 일치시킴)
            message = {
                'nomaly_correct_predictions': self.classification_stats['nomaly_correct_predictions'],
                'nomaly_incorrect_predictions': self.classification_stats['nomaly_incorrect_predictions'],
                'anomaly_correct_predictions': self.classification_stats['anomaly_correct_predictions'],
                'anomaly_incorrect_predictions': self.classification_stats['anomaly_incorrect_predictions']
            }
            
            # 딕셔너리를 JSON 문자열로 변환
            message_json = json.dumps(message)
            
            # 문자열을 전송
            self.send_tcp_message(header, message_json)
            self.file_end = True
            print("파일 종료 메시지 수신 및 처리 완료")
        except Exception as e:
            print("파일 종료 처리 중 오류 발생:", e)

if __name__ == "__main__":
    time_start = time.time()
    FederatedClient(TCP_SERVER_IP, TCP_SERVER_PORT, datasource)