import warnings
warnings.filterwarnings("ignore")

import os
import tensorflow as tf
import psutil
import subprocess

# Check available GPU list
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if not gpus:
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
from datasetLoader import load_synthia_dataset
import datetime,time
import socket
import struct
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import resource  # 메모리 측정을 위한 리소스 모듈 임포트
from tensorflow.keras.utils import to_categorical

print("now is {}".format(datetime.datetime.today()))
import threading
import signal
import sys

num_classes=0
selected_labels=[]
benign_only = False

# TCP client settings
if os.environ.get('CLIENT') is not None:
    TCP_SERVER_IP = 'gateway'  # IP address of the receiving Raspberry Pi
    TCP_SERVER_PORT = 3105
else:
    TCP_SERVER_IP = '192.168.0.10'  # 수신 라즈베리파이의 IP 주소
    TCP_SERVER_PORT = 3105
    
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class LocalModel(object):
    def __init__(self, model_config, num_classes, selected_labels):
        self.model_config = model_config
        self.model = model_from_json(model_config['model_json'])
        
        datasource = load_synthia_dataset(binary=False, object_classes=selected_labels)
        self.x_train, self.y_train, self.x_test, self.y_test = datasource
        self.anomaly_threshold = None
        
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):        
        start_time = time.time()
        self.model.compile(
            loss='categorical_crossentropy',  # 손실 함수 변경
            optimizer=RMSprop(),  # 옵티마이저 변경 가능
            metrics=['accuracy']
        )
        self.loss = self.model.fit(
            self.x_train, self.y_train,  # y_train으로 변경
            epochs=self.model_config['epoch_per_round'],
            batch_size=self.model_config['batch_size'],
            validation_data=(self.x_test, self.y_test),  # y_test로 변경
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

        print("Computing threshold from training data predictions...")
        train_preds = self.model.predict(self.x_train, batch_size=16, verbose=1)
        train_anomaly_scores = calculate_losses(self.x_train, train_preds)
        self.anomaly_threshold = np.percentile(train_anomaly_scores, 99)
        print("Computed threshold:", self.anomaly_threshold)

        testing_set_predictions = self.model.predict(self.x_test, verbose=1)
        test_anomaly_scores = calculate_losses(self.x_test, testing_set_predictions)

        # 이진 장애 탐지를 위한 임계값 적용
        binary_predictions = np.zeros(len(test_anomaly_scores))
        binary_predictions[test_anomaly_scores > self.anomaly_threshold] = 1

        # 원-핫 인코딩된 y_test를 이진 값으로 변환 (예: BENIGN이면 0, 그 외는 1)
        true_labels = np.argmax(self.y_test, axis=1)

        # selected_labels 전역 변수를 사용하여 BENIGN의 인덱스를 찾아 benign_index로 설정
        benign_index = 0
        if selected_labels:
            for idx, lab in enumerate(selected_labels):
                if lab.upper() == "BENIGN":
                    benign_index = idx
                    break

        # BENIGN 클래스(benign_index) 외의 값은 모두 1(attack)로 매핑
        binary_true_labels = np.where(true_labels == benign_index, 0, 1)

        precision = precision_score(binary_true_labels, binary_predictions, average='binary')
        recall = recall_score(binary_true_labels, binary_predictions, average='binary')
        f1 = f1_score(binary_true_labels, binary_predictions, average='binary')
        print("Performance over the testing data set:")
        print("  Recall: {:.16f}, Precision: {:.16f}, F1: {:.16f}".format(recall, precision, f1))
        return f1, precision, recall

# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 1200
    def __init__(self, server_host, server_port, benign_train_only=benign_only):
        self.benign_train_only = benign_train_only
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
        
        # 최초 실행 시간을 기준으로 JSON 파일 생성
        device = "gpu" if tf.config.list_physical_devices("GPU") and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        self.json_file_name = os.path.join("results", device, f"{self.execution_folder}.json")
        if not os.path.exists(os.path.dirname(self.json_file_name)):
            os.makedirs(os.path.dirname(self.json_file_name))
        
        # JSON 파일 초기화
        with open(self.json_file_name, "w") as f:
            json.dump([], f, indent=4)
        
        print(f"데이터가 저장될 폴더: {self.execution_folder}")
        print(f"JSON 파일이 생성되었습니다: {self.json_file_name}")

        # 공격자 클라이언트의 연결을 받을 소켓 설정 (예: 포트 4000 사용)
        # self.setup_attacker_listener(attacker_port=4000)
        # self.attacker_thread = threading.Thread(target=self.receive_attacker_messages, daemon=True)
        # self.attacker_thread.start()

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

    # def setup_attacker_listener(self, attacker_port):
    #     """공격자 클라이언트의 연결을 수신하기 위한 서버 소켓을 설정합니다."""
    #     self.attacker_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     self.attacker_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #     self.attacker_listener.bind(('', attacker_port))
    #     self.attacker_listener.listen(5)
    #     print(f"공격자 패킷 수신을 위한 포트({attacker_port})에서 대기 중입니다.")

    # def receive_attacker_messages(self):
    #     """공격자(attacker) 클라이언트의 메시지를 수신 및 처리합니다.
    #     수신된 메시지는 'classify_packet' 이벤트로 처리되어,
    #     on_classify_packet 함수를 호출합니다.
    #     """
    #     while True:
    #         try:
    #             attacker_conn, attacker_addr = self.attacker_listener.accept()
    #             print("공격자 클라이언트 연결됨:", attacker_addr)
    #             while True:
    #                 # 먼저 4바이트 크기의 메시지 길이를 수신
    #                 message_length_bytes = self.recv_exactly_from(attacker_conn, 4)
    #                 message_length = int.from_bytes(message_length_bytes, byteorder='big')
    #                 print("공격자 메시지 예상 길이:", message_length)
                    
    #                 # 지정된 길이만큼의 데이터를 읽어들임
    #                 json_message = self.recv_exactly_from(attacker_conn, message_length)
    #                 message_data = json.loads(json_message)
                    
    #                 header = message_data['header']
    #                 message = message_data['message']
                    
    #                 # 공격자 클라이언트의 메시지 처리 로직을 여기에 구현합니다.
    #                 self.handle_message(header, message)
    #         except Exception as e:
    #             if not self.file_end:
    #                 print("공격자 메시지 수신 중 오류:", e)
    #                 break

    # def recv_exactly_from(self, conn, size):
    #     """특정 연결(conn)에서 정확히 size 바이트만큼 데이터 수신"""
    #     data = b""
    #     while len(data) < size:
    #         chunk = conn.recv(size - len(data))
    #         if not chunk:
    #             raise ConnectionError("연결이 끊어졌습니다.")
    #         data += chunk
    #     return data

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
        global num_classes, selected_labels
        num_classes = model_config['num_classes']
        selected_labels = model_config['selected_labels']
        # 모델 및 데이터셋 초기화
        self.local_model = LocalModel(model_config, num_classes, selected_labels)
        
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
        global num_classes
        num_classes = req['num_classes']

        self.local_model.model = model_from_json(req['model_json'])
        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])
        with self.eval_lock:
            self.local_model.set_weights(weights)

    def save_round_time(self, round_number, train_time, peak_memory):
        """
        각 라운드의 학습 시간, 최고 메모리 사용량, 라운드 번호, 그리고 실행 장치(CPU 또는 GPU)를
        최초 실행 시간에 기반한 JSON 파일에 저장합니다.
        """
        # 기존 데이터 불러오기
        with open(self.json_file_name, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print("JSON 파일 로드 오류:", e)
                data = []

        # 새 데이터를 추가
        data.append({
            "round_number": round_number,
            "train_time": train_time,
            "peak_memory": peak_memory,
            "device": "gpu" if tf.config.list_physical_devices("GPU") and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        })

        # 파일에 저장
        with open(self.json_file_name, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Round {round_number} 학습 시간 {train_time:.4f}초, 최고 메모리 사용량 {peak_memory:.2f} MB가 {self.json_file_name} 에 저장되었습니다.")

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

    # 저장된 모델을 불러오는 함수 추가
    # def load_saved_model(self):
    #     import os
    #     from tensorflow.keras.models import load_model
    #     model_path = "saved_model.h5"
    #     if os.path.exists(model_path):
    #         print("저장된 모델을 불러옵니다:", model_path)
    #         self.local_model.model = load_model(model_path)
    #     else:
    #         print("저장된 모델이 없습니다. 모델을 먼저 학습시키고 저장해주세요.")

    # def on_classify_packet(self, payload):
    #     global num_classes, selected_labels
    #     if self.local_model is None:
    #         print("로컬 모델이 초기화되지 않았습니다. 먼저 모델을 초기화해주세요.")
    #         return
    #     try:
    #         packet = payload.get("packet")
    #         true_label = payload.get("true_label")
            
    #         packet_array = np.array(packet)
    #         if packet_array.ndim == 1:
    #             packet_array = np.expand_dims(packet_array, axis=0)
    #         packet_array = packet_array.reshape((-1, 78, 1))
            
    #         prediction = self.local_model.model.predict(packet_array, verbose=1)
    #         print(f"Prediction: {prediction}")
    #         predicted_idx = np.argmax(prediction, axis=1)[0]
    #         print(f"Predicted index: {predicted_idx}")
            
    #         class_mapping = {}
    #         for i, label in enumerate(selected_labels):
    #             if label.upper() == "BENIGN":
    #                 class_mapping[i] = "BENIGN"
    #             else:
    #                 class_mapping[i] = "Attack"
    #         predicted_label = class_mapping.get(predicted_idx, "Unknown")
            
    #         result = "nomaly" if predicted_label == "BENIGN" else "anomaly"
    #         is_correct = (predicted_label == true_label) or (true_label.upper() != "BENIGN" and predicted_label == "Attack")
            
    #         print("\n[패킷 분류 결과]")
    #         print(f"  - 실제 라벨: {true_label}")
    #         print(f"  - 예측 라벨: {selected_labels[predicted_idx]}")
    #         print(f"  - 분류 결과 (nomaly/anomaly): {result}")
            
    #         if not hasattr(self, 'classification_stats'):
    #             self.classification_stats = {
    #                 'nomaly_correct_predictions': 0, 
    #                 'anomaly_correct_predictions': 0,
    #                 'nomaly_incorrect_predictions': 0,
    #                 'anomaly_incorrect_predictions': 0,
    #                 'label_stats': {}
    #             }
            
    #         if true_label not in self.classification_stats['label_stats']:
    #             self.classification_stats['label_stats'][true_label] = {'correct': 0, 'incorrect': {}, 'total': 0, 'correct_but_different': {}}
            
    #         self.classification_stats['label_stats'][true_label]['total'] += 1
    #         if is_correct:
    #             self.classification_stats['label_stats'][true_label]['correct'] += 1
    #             if predicted_label != true_label:
    #                 if selected_labels[predicted_idx] not in self.classification_stats['label_stats'][true_label]['correct_but_different']:
    #                     self.classification_stats['label_stats'][true_label]['correct_but_different'][selected_labels[predicted_idx]] = 0
    #                 self.classification_stats['label_stats'][true_label]['correct_but_different'][selected_labels[predicted_idx]] += 1
    #         else:
    #             if result == 'nomaly':
    #                 self.classification_stats['nomaly_incorrect_predictions'] += 1
    #             else:
    #                 self.classification_stats['anomaly_incorrect_predictions'] += 1
    #             if selected_labels[predicted_idx] not in self.classification_stats['label_stats'][true_label]['incorrect']:
    #                 self.classification_stats['label_stats'][true_label]['incorrect'][selected_labels[predicted_idx]] = 0
    #             self.classification_stats['label_stats'][true_label]['incorrect'][selected_labels[predicted_idx]] += 1
            
    #         print("\n[라벨별 예측 결과]")
    #         for label, stats in self.classification_stats['label_stats'].items():
    #             total = stats['total']
    #             correct = stats['correct']
    #             incorrect_total = sum(stats['incorrect'].values())
    #             correct_ratio = (correct / total) * 100 if total > 0 else 0
    #             incorrect_ratio = (incorrect_total / total) * 100 if total > 0 else 0
    #             print(f"\n라벨: {label}")
    #             print(f"  - 총 개수: {total}")
    #             print(f"  - 정답 개수: {correct} ({correct_ratio:.2f}%)")
    #             for pred_label, count in stats['correct_but_different'].items():
    #                 print(f"    - {pred_label}: {count} ({(count / total) * 100:.2f}%)")
    #             print(f"  - 오답 개수: {incorrect_total} ({incorrect_ratio:.2f}%)")
    #             for pred_label, count in stats['incorrect'].items():
    #                 print(f"    - {pred_label}: {count} ({(count / total) * 100:.2f}%)")
        
    #     except Exception as e:
    #         print("분류 중 오류 발생:", e)

    # def on_file_end(self):
    #     try:
    #         header = b'ATTACKS'
    #         message = {
    #             'nomaly_correct_predictions': self.classification_stats['nomaly_correct_predictions'],
    #             'nomaly_incorrect_predictions': self.classification_stats['nomaly_incorrect_predictions'],
    #             'anomaly_correct_predictions': self.classification_stats['anomaly_correct_predictions'],
    #             'anomaly_incorrect_predictions': self.classification_stats['anomaly_incorrect_predictions']
    #         }
            
    #         message_json = json.dumps(message)
    #         self.send_tcp_message(header, message_json)
    #         self.file_end = True
            
    #         print("\n[원본 라벨에 대한 분류 결과]")
    #         for label, stats in self.classification_stats['label_stats'].items():
    #             total = stats['total']
    #             correct = stats['correct']
    #             incorrect_total = sum(stats['incorrect'].values())
    #             correct_ratio = (correct / total) * 100 if total > 0 else 0
    #             incorrect_ratio = (incorrect_total / total) * 100 if total > 0 else 0
    #             print(f"\n라벨: {label}")
    #             print(f"  - 총 개수: {total}")
    #             print(f"  - 정답 개수: {correct} ({correct_ratio:.2f}%)")
    #             for pred_label, count in stats['correct_but_different'].items():
    #                 print(f"    - {pred_label}: {count} ({(count / total) * 100:.2f}%)")
    #             print(f"  - 오답 개수: {incorrect_total} ({incorrect_ratio:.2f}%)")
    #             for pred_label, count in stats['incorrect'].items():
    #                 print(f"    - {pred_label}: {count} ({(count / total) * 100:.2f}%)")
            
    #         print("[파일 종료 메시지 수신 및 처리 완료]")
    #     except Exception as e:
    #         print("파일 종료 처리 중 오류 발생:", e)

if __name__ == "__main__":
    time_start = time.time()
    FederatedClient(TCP_SERVER_IP, TCP_SERVER_PORT)