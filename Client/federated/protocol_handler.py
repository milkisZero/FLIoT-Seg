"""
Federated Learning 프로토콜 핸들러 (원본 메소드 유지)
"""
import json
import time
from model.local_model import LocalModel
from utils.pickle_utils import pickle_string_to_obj, obj_to_pickle_string
from keras.models import model_from_json
import numpy as np
import os
import threading

class FLProtocolHandler:
    """원본 FederatedClient의 프로토콜 핸들러 메소드를 그대로 가져옴"""
    
    def __init__(self, tcp_client, time_start, result_manager,execution_folder):
        self.tcp_client = tcp_client
        self.local_model = None
        self.time_start = time_start
        self.result_manager = result_manager
        self.stop_training = False
        self.execution_folder = execution_folder
        self.eval_lock = threading.Lock()
        self.num_classes = None
        self.selected_labels = None
        # TCP 클라이언트의 메시지 핸들러로 등록
        self.tcp_client.message_handler = self.handle_message
    
    # ========================================
    # 원본 메소드 그대로 복사
    # ========================================
    
    def handle_message(self, header, message):
        """원본 라인 144-164 그대로"""
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
    
    def on_connect(self):
        """원본에서 가져옴"""
        print("Connected to server")
    
    def on_disconnect(self):
        """원본에서 가져옴"""
        print("Disconnected from server")
    
    def on_reconnect(self):
        """원본에서 가져옴"""
        print("Reconnecting to server")
    
    def on_init(self, *args):
        model_config = args[0]
        print("Init message received:", model_config)
        global num_classes, selected_labels
        num_classes = model_config['num_classes']
        selected_labels = model_config['selected_labels']
        # 모델 및 데이터셋 초기화
        self.local_model = LocalModel(model_config, num_classes, selected_labels)
        
        # if self.benign_train_only:
        #     # benign label(0) 데이터만 사용하도록 확인 및 추가 필터링
        #     indices = np.where(self.local_model.y_train == 0)[0]
        #     self.local_model.x_train = self.local_model.x_train[indices]
        #     self.local_model.y_train = self.local_model.y_train[indices]
        #     print("benign_train_only 플래그 활성화: benign 데이터만 학습에 사용합니다.")
        
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
        self.tcp_client.send_tcp_message(header, FL_ready)

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
        self.result_manager.save_round_time(req['round_number'], train_time, peak_memory)
    
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
        self.tcp_client.send_tcp_message(header, resp)
    
        additional_metrics = {
            'round_number': req['round_number'],
            'loss': float(self.local_model.loss.history['loss'][-1]),
            'val_loss': float(self.local_model.loss.history.get('val_loss', [np.nan])[-1]),
            # 학습 에폭 끝난 시점에서의 masked pixel acc (history에는 우리 커스텀 메트릭 이름이 들어갈 수도 있음)
            'masked_pixel_acc': float(self.local_model.loss.history.get('masked_pixel_accuracy', [np.nan])[-1])
        }
        self.tcp_client.send_additional_metrics(additional_metrics)

    def on_eval(self, *args):
        req = args[0]
        event = args[1]
        if(event == 'stop_and_eval'):
            self.stop_training = True
        if req['weights_format'] == 'pickle':
            weights = pickle_string_to_obj(req['current_weights'])
        with self.eval_lock:
            self.local_model.set_weights(weights)
            
        round_number=req["round_number"]
        f1, precision, recall = self.local_model.evaluate1(round_number)
        self.result_manager.save_eval_result(
            round_number=round_number,
            f1=f1,
            precision=precision,
            recall=recall
        )
        
        time_end = time.time()
        print('\033[1;35;0m Time cost = %fs \033[0m' % (time_end - self.time_start))
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
        self.tcp_client.send_tcp_message(header, resp)

        additional_results = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        self.tcp_client.send_additional_results(additional_results)

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
    
    def on_classify_packet(self, payload):
        """패킷 분류 (원본에서 주석 처리됨)"""
        pass
    
    def on_file_end(self):
        """파일 종료 (원본에서 주석 처리됨)"""
        pass