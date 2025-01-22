import warnings
warnings.filterwarnings("ignore")
#author:chunjiong zhang
#date:2020/07/16


from collections import defaultdict
from typing import Dict, List, Type
from keras.models import load_model
from keras.engine import Layer, Model

import pickle#python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
import keras
import uuid#  UUID是128位的全局唯一标识符，通常由32字节的字符串表示。它可以保证时间和空间的唯一性，也称为GUID，全称为：
            #UUID —— Universally Unique IDentifier      Python 中叫 UUID
    #它通过MAC地址、时间戳、命名空间、随机数、伪随机数来保证生成ID的唯一性。
from keras.models import Sequential,Model,Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from keras import backend as K


import msgpack#信息压缩
import random
import codecs#Python中用codecs处理各种字符编码的文件
import numpy as np
import json
import msgpack_numpy#信息编码格式
# https://github.com/lebedov/msgpack-numpy
import logging
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from keras.layers.core import Lambda

from keras.models import load_model


logging.basicConfig(
    level=logging.INFO,
)

import sys
#from ranger import Ranger
import time
from sklearn.metrics.pairwise import cosine_similarity
from flask import *
from flask_socketio import SocketIO#SocketIO是大名鼎鼎的实时通讯库,可以在服务器和页面之间轻松的实现双向实时通讯, 兼容性好,使用方便.多用来制作聊天/直播室, 等需要实时传输数据的地方.
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/
import re
import requests
import time
import threading

MOBIUS_URL = "http://192.168.0.60:7579"
HEADERS = {
    "X-M2M-Origin": "SOrigin",
    "X-M2M-RI": "12345",
    "Content-Type": "application/json"
}

SUB_CNT_List = ['client1FromC', 'client2FromC']

# 컨테이너 이름 , 이름/set == 토픽
PUB_CNT_List = ['client1FromS', 'client2FromS']

HOST = "192.168.0.60"
PORT = 5011

class GlobalModel(object):#类文档字符串
    """docstring for GlobalModel"""
    def __init__(self):#__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法，当创建了这个类的实例时就会调用该方法
        self.model = self.build_model()#self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
        self.current_weights = self.model.get_weights()
        # for convergence check
        self.prev_train_loss = None

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.training_start_time = int(round(time.time()))#以cluster节点的时间为基准
    
    # def build_model(self):
    #     raise NotImplementedError()#raise可以实现报出错误的功能,而此时产生的问题分类是NotImplementedError。
    #
    # # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):#全局模型参数更新（可修改）
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)


        for c in range(len(client_weights)):
            print("client_weights",len(client_weights))
            #print("client_weights_shape",client_weights[c].shape)
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        self.current_weights = new_weights        

    def aggregate_loss_accuracy(self, client_losses, client_sizes):
        total_size = np.sum(client_sizes)
        print("-----client number:-------",client_sizes)
        print('\033[1;35;0m client_sizes \033[0m', client_sizes)
        print('\033[1;35;0m client_losses \033[0m',client_losses)  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')


        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_loss
    # cur_round coule be None    , cur_round
    def aggregate_train_loss_accuracy(self, client_losses,client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss = self.aggregate_loss_accuracy(client_losses, client_sizes)
        self.train_losses += [[cur_round,cur_time, aggr_loss]]
        #self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies
        }

class GlobalModel_KDD_AE(GlobalModel):
    def __init__(self):
        super(GlobalModel_KDD_AE, self).__init__()

    def build_model(self):
        # ~35MB worth of parameters
        #怎么获得模型大小
        # input数据接口
        input_img = Input(shape=(118,))
        x = Dense(64, activation='relu', kernel_initializer='random_uniform',name='encoded1')(input_img)
        x = Dense(32, activation='tanh',  name='encoded2')(x)
        x = Dense(12, activation='tanh',  name='encoded3')(x)
        x = Dense(118, activation=None,  name='net0_decoded4')(x)
        model = Model(inputs=input_img, outputs=x)
        model.summary()
        for layer in model.layers:
            print(layer.name)

        #adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
        adam = keras.optimizers.Adam(lr = 0.007, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
        # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
        # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
        model.compile(loss=keras.losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
        return model

class FLServer(object):
    MIN_NUM_WORKERS = 2#最少节点数量设置
    MAx_NUM_ROUNDS = 6#设定联邦循环次数
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1#设置节点数量，作用，多少比例的掉队。
    ROUNDS_BETWEEN_VALIDATIONS = 2

    def __init__(self, global_model, host, port):
        self.global_model = global_model()
        self.ready_client_sids = set()
        self.model_id = str(uuid.uuid4())#uuid4()——基于随机数,由伪随机数得到，有一定的重复概率，该概率可以计算出来。
      
        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        #####
        
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.add_url_rule('/notify', 'notify', self.notify, methods=['POST'])
        self.app.add_url_rule('/aeWatcher', 'aeWatcher', self.aeWatcher, methods=['POST'])
        self.app.add_url_rule('/aeSub', 'aeSub', self.aeSub, methods=['POST'])  
                
        @self.app.route('/')
        def dashboard():
            """测试页面"""
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())        
        
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
    def create_cntSub(self,path):
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

    def create_cntPub(self):
        # pub 컨테이너 이름 == conf의 sub, 키 이름
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
        
        
            # payload = {
            #     "m2m:sub": {
            #         "rn": "FLserverSub",  # 구독 이름
            #         "nu": [f"mqtt://192.168.0.60:1883/SFLIoT/{e}/set?ct=json"],  # 알림을 받을 서버 URL
            #         "nct": 2,  # 알림 내용 형식 (전체 콘텐츠)
            #         "enc": {
            #             "net": [1,2,3,4]  # 이벤트 조건: 데이터 생성
            #         },
            #         "exc": 100  # 최대 알림 횟수
            #     }
            # }
            
            # url = MOBIUS_URL + '/Mobius/FLIoT/' + e
            # headers = HEADERS
            # headers["Content-Type"] = "application/json; ty=23"

            # try:
            #     response = requests.post(url, headers=headers, json=payload)

            #     if response.status_code == 201:
            #         print("sub created successfully!")
            #     else:
            #         print(f"Failed to create sub: {response.status_code}, {response.text}")

            # except Exception as e:
            #     print(f"Error creating sub: {e}")

    def publish(self, path, data):
        payload = {
            "m2m:cin": {
                "con": data,   
            }
        }
                
        url = MOBIUS_URL + '/Mobius/FLIoT/' + path
        headers = HEADERS
        headers["Content-Type"] = "application/json; ty=4"

        try:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 201:
                print("Publish successfully!")
            else:
                print(f"Failed to Publish: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Error creating cntPub: {e}")
    
    # aeWatcher 알림 엔드포인트
    def aeWatcher(self):
        try:
            data = request.json
            print("Received AE:")

            if "m2m:sgn" in data:
                # rn인지 ri인지 선택해야 함
                content = data["m2m:sgn"]["nev"]["rep"]["m2m:ae"]["rn"]
                print(f"New AE detected(rn): {content}")
                self.create_aeSub(content)
                self.create_cntPub()
                
            return jsonify({"status": "received"}), 200

        except Exception as e:
            print(f"Error processing notification: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    # aeSub 알림 엔드포인트
    def aeSub(self):
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
                    self.create_cntSub(url + content)
                
            return jsonify({"status": "received"}), 200

        except Exception as e:
            print(f"Error processing notification: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    # 데이터 엔드포인트
    def notify(self):
        try:
            # Mobius에서 전송된 알림 데이터
            data = request.json
            print("Received Notification:")

            # 알림 데이터에서 콘텐츠 추출
            if "m2m:sgn" in data:
                content = data["m2m:sgn"]["nev"]["rep"]["m2m:cin"]["con"]
                print("Updated Content: ", content)   
                
                url = data["m2m:sgn"]['sur']
                client_id = url.split('/')[2].replace("FromC", "")
                self.on_message(content, client_id)

            return jsonify({"status": "received"}), 200

        except Exception as e:
            print(f"Error processing notification: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # def on_connect(self, userdata, flags, connection):
    #     if connection == 0:
    #         print(f"Connected")
    #     else:
    #         print(f"Failed to connect, return code {connection}")

    # def on_disconnect(self):
    #     print("Disconnected from MQTT Broker")

    def on_message(self, msg, client_id):
        payload = json.dumps(msg);
        
        try:
            # JSON 처리 시도
            payload = json.loads(payload)
            print("Detected JSON format:", payload)
        except Exception as e:
            print(f"on_message Error: {e}")
            # try:
            #     # Pickle 처리 시도
            #     payload = pickle.loads(msg.payload)
            # #  print("Detected Pickle format:", payload)
            #     print(payload)
            # except Exception as e:
            #     print(f"Failed to process message: {e}")
     
        if client_id:
            self.register_handles(payload, client_id)       
        else :
            print(payload)
             
    def register_handles(self, payload, client_id):
        # single-threaded async, no need to lock
        
        event = payload['event']
            
        if event == 'connect':
            print(client_id, "connected")# # request.sid,,,io客户端的sid, socketio用此唯一标识客户端.
            print('\033[1;35;0m connected \033[0m')
            self.ready_client_sids.add(client_id)
            
        elif event == 'reconnect':
            print(client_id, "reconnect")
            self.ready_client_sids.add(client_id)
            
        elif event == 'disconnect':
            print(client_id, "disconnected")
            if client_id in self.ready_client_sids:
                self.ready_client_sids.remove(client_id)
                
        elif event == 'client_wake_up':
            print("client wake_up: ", client_id)
            data = {
                'event': 'init', 
                'payload': {
                    'model_json': self.global_model.model.to_json(),
                    'model_id': self.model_id,

                    #'data_split': (0.6, 0.3, 0.1), # train, test, valid
                    'epoch_per_round': 1,
                    'batch_size': 100
                }
            }
            self.publish(client_id+'FromS', data)
                        
        elif event == 'client_ready':
            data = payload['payload']
            print("client ready for training", client_id, data)
            self.ready_client_sids.add(client_id)
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:
                self.train_next_round()
                
        elif event == 'client_update':
            data = payload['payload']
            print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", client_id)
            print('\033[1;35;0m handle client_update \033[0m')
            for x in data:
                if x != 'weights':
                    print(x, data[x])

            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])

                self.global_model.update_weights(
                    [x['weights'] for x in self.current_round_client_updates],
                    [x['train_size'] for x in self.current_round_client_updates],
                )
                aggr_train_loss = self.global_model.aggregate_train_loss_accuracy(
                    [x['train_loss'] for x in self.current_round_client_updates],
                    # [x['train_accuracy'] for x in self.current_round_client_updates],
                    [x['train_size'] for x in self.current_round_client_updates],
                    self.current_round
                )

                print("aggr_train_loss", aggr_train_loss)
                # print("aggr_train_accuracy", aggr_train_accuracy)

                if self.global_model.prev_train_loss is not None and \
                        (
                                self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss < .1:  # 我修改了
                    # converges
                    print("converges! starting test phase..")  # 判断收敛性
                    print(
                        '\033[1;35;0m converges! starting test phase.. \033[0m')  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')
                    self.stop_and_eval()

                self.global_model.prev_train_loss = aggr_train_loss

                if self.current_round >= FLServer.MAx_NUM_ROUNDS:
                    self.stop_and_eval()
                else:
                    self.train_next_round()
                    
        elif event == 'client_eval':
            data = payload['payload']
            if self.eval_client_updates is None:
                return
            print("handle client_eval", client_id)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            print('\033[1;35;0m == done == \033[0m')  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')
            time_end = time.time()
            print('totally cost', time_end - time_start)

            self.eval_client_updates = None  # special value, forbid evaling again

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        print("### Round ", self.current_round, "###")
        client_sids_selected = random.sample(list(self.ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)#为了提取出N个不同元素的样本用来(所有内容，需要的数量)
        print("request updates from", client_sids_selected)

        # by default each client cnn is in its own "room"
        for rid in client_sids_selected:
            data = {
                'event': 'request_update', 
                'payload' : {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),

                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }
            }
            self.publish(rid+'FromS', data)

    def stop_and_eval(self):
        #self.global_model.save("global_model.h5")
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            data = {
                'event': 'stop_and_eval',
                'payload':  {
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle'
                }
            }
            self.publish(rid, data)

    def start_flask(self):
        print(f"Starting Flask server at {self.host}:{self.port}...")
        self.app.run(host=self.host, port=self.port)

    def start(self):
        flask_thread = threading.Thread(target=self.start_flask)
        flask_thread.daemon = True  # 메인 스레드가 종료되면 Flask 스레드도 종료
        flask_thread.start()
        
        time.sleep(1) 
        self.create_aeWatcher()

        # 서버 실행 유지
        while True:
            try:
                time.sleep(1)  # 메인 루프에서 대기
            except KeyboardInterrupt:
                print("Shutting down FLServer...")
                break

def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)

if __name__ == '__main__':
    time_start = time.time()
    server = FLServer(GlobalModel_KDD_AE, HOST, PORT)
    print("listening on ...")
    server.start()
    


    