import warnings
warnings.filterwarnings("ignore")
#author:chunjiong zhang
#date:2020/07/16


from collections import defaultdict
from typing import Dict, List, Type
from keras.models import load_model

# from keras.engine import Layer, Model # Raspberry yxyao 20241224
from tensorflow.keras.layers import Layer # Raspberry yxyao 20241224
from tensorflow.keras.models import Model # Raspberry yxyao 20241224

import pickle#python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
import keras
import uuid#  UUID是128位的全局唯一标识符，通常由32字节的字符串表示。它可以保证时间和空间的唯一性，也称为GUID，全称为：
            #UUID —— Universally Unique IDentifier      Python 中叫 UUID
    #它通过MAC地址、时间戳、命名空间、随机数、伪随机数来保证生成ID的唯一性。

# from keras.models import Sequential,Model,Input # Raspberry yxyao 20241224
from tensorflow.keras.models import Sequential, Model # Raspberry yxyao 20241224
from tensorflow.keras.layers import Input # Raspberry yxyao 20241224

from keras.layers import Dense, Dropout, Flatten

# from keras.layers.advanced_activations import LeakyReLU # Raspberry yxyao 20241224
from tensorflow.keras.layers import LeakyReLU # Raspberry yxyao 20241224

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

import paho.mqtt.client as mqtt
import re

MQTT_BROKER_HOST  = '192.168.0.60'
MQTT_BROKER_PORT  = 1883
endpoint =  ''
# connectUrl = f"mqtt://{host}:${port}${endpoint}"


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

######## Flask server with Socket IO ########

# Federated Averaging algorithm with the server pulling from clients


topics = ["/thyme/weights1",
          "/thyme/weights2",
          "/thyme/metrics1",
          "/thyme/metrics2",
          "/thyme/results1",
          "/thyme/results2",
          "/thyme/client1",
          "/thyme/client2",
          ]

class FLServer(object):

    MIN_NUM_WORKERS = 2#最少节点数量设置
    MAx_NUM_ROUNDS = 6#设定联邦循环次数
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1#设置节点数量，作用，多少比例的掉队。
    ROUNDS_BETWEEN_VALIDATIONS = 2

    def __init__(self, global_model, host, port, mqtt_broker, mqtt_port):
        self.global_model = global_model()
        self.ready_client_sids = set()
        self.app = Flask(__name__)
    
        self.model_id = str(uuid.uuid4())#uuid4()——基于随机数,由伪随机数得到，有一定的重复概率，该概率可以计算出来。
      
        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        #####
        
        @self.app.route('/')
        def dashboard():
            """测试页面"""
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())
        
        # MQTT 브로커 연결 설정
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "PythonMQTTClient" )
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        self.register_handles()
        
    # MQTT 콜백 함수 정의
    def on_connect(self, mqtt_client, userdata, flags, connection):
        if connection == 0:
            print(f"Connected to MQTT Broker: {host}:{port}")
            # 연결 성공 후 토픽 구독
            
            for topic in topics:
                mqtt_client.subscribe(topic)
                print(f"Subscribed to topic: {topic}")
        else:
            print(f"Failed to connect, return code {connection}")

    def on_message(self, mqtt_client, userdata, msg):
        payload = None;
        try:
            # JSON 처리 시도
            payload = json.loads(msg.payload.decode('utf-8'))
            print("Detected JSON format:", payload)
        # print(len(payload))
        except Exception:
            try:
                # Pickle 처리 시도
                payload = pickle.loads(msg.payload)
            #  print("Detected Pickle format:", payload)
                print(len(payload))
            except Exception as e:
                print(f"Failed to process message: {e}")
        
        
        pattern = r"/thyme/(client\d+)" 
        client_id = re.match(pattern, msg.topic)
                
        #### 웨이트, 매트릭은 client_id가 전송 되지 않음. 굳이 따로 할 필요가 ?
        if client_id:
            self.register_handles(payload, client_id)       
        else :
            print(payload)
             
    def on_disconnect(self):
        print("Disconnected from MQTT Broker")

    def register_handles(self, payload, client_id):
        # single-threaded async, no need to lock
        
        event = payload.event
        
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
            self.mqtt_client.publish(client_id+'/set', data)
            
        elif event == 'client_ready':
            print("client ready for training", client_id, data)
            self.ready_client_sids.add(client_id)
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:
                self.train_next_round()
                
        elif event == 'client_update':
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
            self.mqtt_client.publish(rid+'/set', data)

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
            self.mqtt_client.publish(rid+'/set', data)

    def start(self):
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()
        self.app.run(host=self.host, port=self.port)



def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    time_start = time.time()
    server = FLServer(GlobalModel_KDD_AE, "127.0.0.1", 5011, MQTT_BROKER_HOST, MQTT_BROKER_PORT)
    print("listening on ...");
    server.start()


    