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

        # for i in range(len(new_weights)):
        #     new_weights[i] += client_weights[0][i]
        #     node_weights1 = new_weights
        #     new_weights[i] += client_weights[1][i]
        #     node_weights2 = new_weights
        #node_weights = new_weights
        node_weights1= np.array(client_weights[0]).reshape(1, -1)
        node_weights2 = np.array(client_weights[1]).reshape(1, -1)
        #print(node_weights1[0][0][0])

        #节点间预先相似度计算

    #         """ 计算两个向量x和y的余弦相似度 """

        node1 = np.array(node_weights1[0][0][0]).reshape(1, -1)
        node2 = np.array(node_weights2[0][0][0]).reshape(1, -1)
        cosine1=[]

        cosine=cosine_similarity(node1, node2)
        cosine1.append(cosine[0][0])
        print('\033[1;35;0m cosine_similarity \033[0m', cosine)  # 1.0
        np.savetxt("cosine_similarity.txt",cosine1)
        #if cosine1>0.8:


        for c in range(len(client_weights)):
            print("client_weights",len(client_weights))
            #print("client_weights_shape",client_weights[c].shape)
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        #new_weights[1]=new_weights[0]
        self.current_weights = new_weights 

    def aggregate_loss_accuracy(self, client_losses, client_sizes):
        total_size = np.sum(client_sizes)
        print("-----client number:-------",client_sizes)
        print('\033[1;35;0m client_sizes \033[0m', client_sizes)
        print('\033[1;35;0m client_losses \033[0m',client_losses)  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')


        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))

        #拉格朗日对偶函数构造
        f11=client_losses[0]
        f12= client_losses[1]
        f13 = client_losses[2]

        x1 = [[f11], [f12], [f13]]

        # 标准化处理，因为x1的幅度特别大，求解难寻优
        x1 = MinMaxScaler(feature_range=(0.003, 0.005), copy=True).fit_transform(x1)
        print(x1)

        f11 = x1[0]
        f12 = x1[1]
        f13 = x1[2]
        print(f11, f12, f13)
        from scipy.optimize import minimize
        import math
        x0 = np.asarray((0.33, 0.33, 0.33))

        def fun(args):
            l11, l12, l13 = args
            lambda1 = 0
            for n in range(20):
                v = lambda x0: x0[0] * l11 + x0[1] * l12 + x0[2] * l13 + lambda1 * (
                            x0[0] + x0[1] + x0[2] - 1);
                rho = math.pow(0.8, n)
                lambda1 = lambda1 + rho * (x0[0] + x0[1] + x0[2] - 1)
            return v

        # subgradient 计算  (2+x1)/(1+x2) - 3*x1+4*x3 的最小值  x1,x2,x3的范围都在0.1到0.9 之间
        def con(args1):
            # 约束条件 分为eq 和ineq
            # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
            x1min, x1max, x2min, x2max, x3min, x3max= args1
            cons = ({'type': 'ineq', 'fun': lambda x0: x0[0] - x1min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[0] + x1max},
                    {'type': 'ineq', 'fun': lambda x0: x0[1] - x2min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[1] + x2max},
                    {'type': 'ineq', 'fun': lambda x0: x0[2] - x3min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[2] + x3max},
                    #{'type': 'ineq', 'fun': lambda x0: x0[3] - x4min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[3] + x4max},
                    {'type': 'eq', 'fun': lambda x0: x0[0] + x0[1] + x0[2] - 1})
            return cons

        # 定义常量值
        args = (f11, f12, f13)  # a,b,c,d
        # 设置参数范围/约束条件
        args1 = (0.18, 0.99, 0.015, 0.99, 0.0152, 0.99)  # x1min, x1max, x2min, x2max
        # cons = con(args1)
        # 设置初始猜测值
        res = minimize(fun(args), x0, method='SLSQP', constraints=con(args1))  #
        print("f1 min", res.fun)
        print(res.success)
        # print(res.x)
        p1, p2, p3= res.x
        print("p1, p2, p3", p2, p1, p3)

        #这个位置修改loss 函数

        # aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
        #         for i in range(len(client_sizes)))
        return aggr_loss,res.x

    # cur_round coule be None    , cur_round
    def aggregate_train_loss_accuracy(self, client_losses,client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, p = self.aggregate_loss_accuracy(client_losses, client_sizes)
        self.train_losses += [[cur_round,cur_time, aggr_loss]]
        #self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss,p

    # cur_round coule be None
    # def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
    #     cur_time = int(round(time.time())) - self.training_start_time
    #     aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
    #     self.valid_losses += [[cur_round, cur_time, aggr_loss]]
    #     self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
    #     with open('stats.txt', 'w') as outfile:
    #         json.dump(self.get_stats(), outfile)
    #     return aggr_loss, aggr_accuraries

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies
        }

class GlobalModel_KDD_AE(GlobalModel):
    #余弦相似度构造
    def cosine(self,x1, x2):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_

        output_shape = (1,)
        value = Lambda(_cosine, output_shape=output_shape)([x1, x2])
        return value

    #欧式距离构造
    def relative_euclid(self,x, x_rec):
        def _cosine(x):
            min_val = 1e-3
            def euclid_norm(x):
                return K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))  # tf.reduce_sum
            x_l2 = euclid_norm(x[0])
            res_l2 = euclid_norm(x[0] - x[1])
            return res_l2 / (x_l2 + min_val)
        output_shape = (1,)
        relative_euclid = Lambda(_cosine, output_shape=output_shape)([x, x_rec])  #
        return relative_euclid

    def __init__(self):
        super(GlobalModel_KDD_AE, self).__init__()
        #其中的super类的作用是继承的时候，
        # 调用含super的各个的基类__init__函数，
        # 如果不使用super，就不会调用这些类的__init__函数，
        # 除非显式声明。而且使用super可以避免基类被重复调用

    def build_model(self):
        # ~35MB worth of parameters
        #怎么获得模型大小
        # input数据接口
        input_img = Input(shape=(78,))
        # # 分支0
        x = Dense(64, activation='relu', kernel_initializer='random_uniform',
                  activity_regularizer=regularizers.l2(10e-5),
                  name='encoded1')(input_img)
        # x=Dropout(0.5)(x)
        x = Dense(32, activation='relu', kernel_initializer='random_uniform', name='net0_encoded2')(x)
        x = Dense(16, activation='relu', kernel_initializer='random_uniform', name='net0_encoded3')(x)
        z0 = Dense(1, kernel_initializer='random_uniform', name='net_encodedz')(x)
        x = Dense(16, activation='relu', kernel_initializer='random_uniform', name='net0_decoded1')(x)
        x = Dense(32, activation='relu', kernel_initializer='random_uniform', name='net0_decoded2')(x)
        x = Dense(64, activation='relu', kernel_initializer='random_uniform', name='net0_decoded3')(x)
        x = Dense(78, activation=None, kernel_initializer='random_uniform', name='net0_decoded4')(x)
        #
        cosine0 = self.cosine(input_img, x)
        relative_euclid0 = self.relative_euclid(input_img, x)
        z0 = keras.layers.concatenate([cosine0, relative_euclid0, z0], axis=1)

        # # 分支1
        tower1 = Dense(78, activation='relu', kernel_initializer='random_uniform', name='net1_encoded0')(input_img)
        # tower1 = Dropout(0.5)(tower1)
        tower1 = Dense(112, activation='relu', kernel_initializer='random_uniform', name='net1_encoded1')(tower1)
        tower1 = Dense(81, activation='relu', kernel_initializer='random_uniform', name='net1_encoded2')(tower1)
        tower1 = Dense(55, activation='relu', kernel_initializer='random_uniform', name='net1_encoded3')(tower1)
        tower1 = Dense(28, activation='relu', kernel_initializer='random_uniform', name='net1_encoded4')(tower1)
        z1 = Dense(1, kernel_initializer='random_uniform', name='net1_encodedz')(tower1)
        tower1 = Dense(28, activation='relu', kernel_initializer='random_uniform', name='net1_decoded0')(tower1)
        tower1 = Dense(55, activation='relu', kernel_initializer='random_uniform', name='net1_decoded1')(tower1)
        tower1 = Dense(81, activation='relu', kernel_initializer='random_uniform', name='net1_decoded2')(tower1)
        tower1 = Dense(112, activation='relu', kernel_initializer='random_uniform', name='net1_decoded3')(tower1)
        tower1 = Dense(78, activation=None, kernel_initializer='random_uniform', name='net1_decoded4')(tower1)

        cosine1 = self.cosine(input_img, tower1)
        relative_euclid1 = self.relative_euclid(input_img, tower1)
        z1 = keras.layers.concatenate([cosine1, relative_euclid1, z1], axis=1)

        # # 分支2
        tower2 = Dense(78, activation='relu', kernel_initializer='random_uniform', name='net2_encoded0')(input_img)
        # tower2 = Dropout(0.5)(tower2)
        tower2 = Dense(60, activation='relu', kernel_initializer='random_uniform', name='net2_encoded1')(tower2)
        tower2 = Dense(30, activation='relu', kernel_initializer='random_uniform', name='net2_encoded2')(tower2)
        tower2 = Dense(10, activation='relu', kernel_initializer='random_uniform', name='net2_encoded3')(tower2)
        z2 = Dense(1, kernel_initializer='random_uniform', name='net2_encoded4')(tower2)
        tower2 = Dense(10, activation='relu', kernel_initializer='random_uniform', name='net2_decoded0')(tower2)
        tower2 = Dense(30, activation='relu', kernel_initializer='random_uniform', name='net2_decoded1')(tower2)
        tower2 = Dense(60, activation='relu', kernel_initializer='random_uniform', name='net2_decoded2')(tower2)
        tower2 = Dense(78, activation=None, kernel_initializer='random_uniform', name='net2_decoded3')(tower2)

        # hidden2 = [78, 60, 30, 1, 30, 60, 78]
        # tower2 = input_img
        # n_layer = 0
        # for nums in hidden2:
        #     tower2 = Dense(nums, activation='elu' if nums != 1 else None, name="net2_{}".format(n_layer))(tower2)
        #     #tower2=LeakyReLU(alpha=0.2)(tower2)
        #     #tower2 = Dropout(0.5)(tower2)
        #     n_layer += 1
        #     if nums == 1:
        #         z2 = tower2

        cosine2 = self.cosine(input_img, tower2)
        relative_euclid2 = self.relative_euclid(input_img, tower2)
        z2 = keras.layers.concatenate([cosine2, relative_euclid2, z2], axis=1)

        # 拼接output
        loss0 = keras.losses.mean_squared_error(y_pred=input_img, y_true=x)
        loss1 = keras.losses.mean_squared_error(y_pred=input_img, y_true=tower1)
        loss2 = keras.losses.mean_squared_error(y_pred=input_img, y_true=tower2)
        # seclect minimum loss
        recons22 = keras.backend.maximum(loss0, loss1)
        loss = keras.backend.maximum(recons22, loss2)
        # loss最小的重建误差
        if loss0 == loss:
            tower = x
        elif loss1 == loss:
            tower = tower1
        else:
            tower = tower2
        output = keras.layers.concatenate([z0, z1, z2, tower], axis=1)
        # # GAN
        # output1 = Dense(8, activation='relu', kernel_initializer='random_uniform',
        #                 activity_regularizer=regularizers.l2(10e-5), name='gan1')(output)
        # output1 = Dense(78, activation='relu', kernel_initializer='random_uniform',
        #                 activity_regularizer=regularizers.l2(10e-5), name='gan2')(output1)
        # output1 = Dense(64, activation='relu', kernel_initializer='random_uniform',name='encoded7')(output1)

        # 把前面的计算逻辑，分别指定input和output，并构建成网络
        model = Model(inputs=input_img, outputs=tower)
        model.summary()
        for layer in model.layers:
            print(layer.name)


        #model = load_model('model_raw.h5')
        # 编译model
        #adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
        adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
        # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
        # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
        model.compile(loss=keras.losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
        #model.save("global_m.h5")

        # for layer in model.layers:
        #     print(layer.name)
        # #模型压缩
        # model = compress(model, 7e-1)
        return model

class FLServer(object):
    MIN_NUM_WORKERS = 3#最少节点数量设置
    MAx_NUM_ROUNDS = 10#设定联邦循环次数
    NUM_CLIENTS_CONTACTED_PER_ROUND = 3#设置节点数量，作用，多少比例的掉队。
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
                #print("Updated Content: ", content)   
                
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
           # print("Detected JSON format:", payload)
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
            print("none exist client_id!!")
             
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
            # data:
            #   weights
            #   train_size
            #   valid_size
            #   train_loss
            #   train_accuracy
            #   valid_loss?
            #   valid_accuracy?

            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                
                # tolerate 30% unresponsive clients
                if len(self.current_round_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )
                    aggr_train_loss, p = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        #[x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )

                    print("aggr_train_loss", aggr_train_loss)
                    #print("aggr_train_accuracy", aggr_train_accuracy)


                    # if self.global_model.prev_train_loss is not None and \
                    #         (self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss < .1:#我修改了
                    #     # converges
                    #     print("converges! starting test phase..")#判断收敛性
                    #     print('\033[1;35;0m converges! starting test phase.. \033[0m')  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')
                    #     self.stop_and_eval()
                    #     return
                    
                    self.global_model.prev_train_loss = aggr_train_loss

                    if self.current_round >= FLServer.MAx_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round(p)
                    
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
    def train_next_round(self,p):
        if p is None:
            p=[1,1,1]#防止在handle_client_ready(data) 中的self.train_next_round(None)出现NONE
        p = [1, 1, 1]#不是multidomian learning，不需要调整p，所以p都设置为1
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
                    'p1': p[0],
                    'p2': p[1],
                    'p3': p[2],
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
            self.publish(rid+'FromS', data)

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
    print("listening on ... {HOST}:{PORT}")
    server.start()
    


    