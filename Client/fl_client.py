import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # -1 to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = filter out INFO, 2 = WARNING, 3 = ERROR

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

print("now is {}".format(datetime.datetime.today()))
datasource= gen_train_valid_data()
import threading

# TCP 클라이언트 설정
if os.environ.get('CLIENT') is not None:
    TCP_SERVER_IP = 'gateway'  # 수신 라즈베리파이의 IP 주소
    TCP_SERVER_PORT = 3105
else:
    TCP_SERVER_IP = '192.168.0.10'  # 수신 라즈베리파이의 IP 주소
    TCP_SERVER_PORT = 3105

class LocalModel(object):
    def __init__(self, model_config, data_collected):
        self.model_config = model_config
        self.model = model_from_json(model_config['model_json'])
        self.x_train, self.y_train, self.x_test, self.y_test = data_collected

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.RMSprop(),
                           metrics=['accuracy'])
        self.loss = self.model.fit(
            self.x_train, self.x_train,
            epochs=self.model_config['epoch_per_round'],
            batch_size=self.model_config['batch_size'],
            validation_data=(self.x_test, self.x_test),
            verbose=1
        )
        current_loss = self.loss.history['loss'][0]
        print('One round training loss: {:.16f}'.format(current_loss))
        return self.model.get_weights(), current_loss

    def evaluate1(self):
        def calculate_losses(x, preds):
            losses = np.zeros(len(x))
            for i in range(len(x)):
                losses[i] = np.mean(np.square(preds[i] - x[i]))
            return losses

        # Always compute threshold using the current training set predictions.
        print("Computing threshold from training data predictions...")
        train_preds = self.model.predict(self.x_train, verbose=1)
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
    def __init__(self, server_host, server_port, datasource):
        self.local_model = None
        self.datasource = datasource
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.tcp_socket.connect((server_host, server_port))
        except Exception as e:
            print("TCP 소켓 연결 실패:", e)
            # 재연결 로직 추가
        
        self.eval_lock = threading.Lock()
        self.socket_lock = threading.Lock()
        
        # TCP 메시지 수신을 별도의 쓰레드에서 실행하여 메인 쓰레드가 블로킹되지 않도록 합니다.
        self.tcp_receive_thread = threading.Thread(target=self.receive_tcp_messages, daemon=True)
        self.tcp_receive_thread.start()
        
        self.test_interval = 60
        # 지속적인 테스트 평가를 위한 쓰레드 실행
        self.testing_thread = threading.Thread(target=self.continuous_testing, daemon=True)
        self.testing_thread.start()

        print("sent wakeup")
        message = json.dumps({
            'event': 'client_wake_up'
        })
        self.send_tcp_message('OPERATE', message)

    def continuous_testing(self):
        while True:
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
        while True:
            try:
                message_length_bytes = self.tcp_socket.recv(4)
                message_length = int.from_bytes(message_length_bytes, byteorder='big')
                print('sum : ', message_length)
                #json_message = self.tcp_socket.recv(65535).decode('utf-8')

                json_message = b""
                while len(json_message) != message_length:
                    chunk = self.tcp_socket.recv(1024)
                    if not chunk:
                        print("연결이 끊어졌습니다.")
                        break
                    json_message += chunk
      #          print(json_message)
                message_data = json.loads(json_message)
     #           print(message_data)
                header = message_data['header']
                message = message_data['message']
                # print(header)
                # print(message)
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
                self.on_eval(message['payload'])
            elif event == 'global_update':
                self.on_global_update(message['payload'])
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
        # 추가적인 초기화 로직
        self.local_model = LocalModel(model_config, self.datasource)

        header = b'OPERATE'
        FL_ready = json.dumps({
                'event': 'client_ready',
                'payload': {
                    'train_size': self.local_model.x_train.shape[0],
                    #'class_distr': my_class_distr  # for debugging, not needed in practice
                }
            })
        # ready to be dispatched for training
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

    def on_request_update(self, *args):
        req = args[0]
        print("update requested")
        print('round_number:', req['round_number'])

        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])

        with self.eval_lock:
            self.local_model.set_weights(weights)
        my_weights, train_loss = self.local_model.train_one_round()

        header = b'OPERATE'
        resp = json.dumps({
            'event': 'client_update',
            'payload': {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'train_loss': train_loss,
            }
        })

        # 피클 파일 생성 및 전송
        filename = f"weights_round_{req['round_number']}.pkl"
        self.send_blob_data(my_weights, filename)

        # SocketIO를 사용하여 'client_update' 이벤트 전송
        self.send_tcp_message(header, resp)

        # 추가 메트릭을 TCP 소켓을 통해 192.168.0.10:3105로 전송
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

if __name__ == "__main__":
    time_start = time.time()
    FederatedClient(TCP_SERVER_IP, TCP_SERVER_PORT, datasource)
