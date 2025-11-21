import threading
import time
from flask import Flask
from communication.mobius_handler import MobiusHandler
from communication.mobius_routes import create_mobius_routes
from communication.api_routes import create_api_routes
from federated.protocol_handler import FLProtocolHandler
from models.unet_model import UNetGlobalModel, UNetLite
from models.DeepLabV3PlusMobileNet import DeepLabV3PlusMobileNet
import json
import time
from utils.gpu_setup import setup_gpu
from federated.result_manager import FLResultManager
import tensorflow as tf
import os
import datetime

class FLServer:
    def __init__(self, global_model, config):
        
        gpu_id = 0 # if gpu_id == -1, use cpu
        setup_gpu(gpu_id=gpu_id, enable_mixed_precision=False)
    
        self.host = config.host
        self.port = config.port
        
        # 글로벌 모델
        self.global_model = global_model(config=config)
        
        device = "gpu" if tf.config.list_physical_devices("GPU") and gpu_id != -1 else "cpu"
        self.execution_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.execution_folder):
            os.makedirs(self.execution_folder)
        self.result_manager = FLResultManager(base_dir= "results", execution_folder = self.execution_folder, device=device)
        
        self.mobius_handler = MobiusHandler(config)
        
        # 연합학습 프로토콜 핸들러
        self.protocol_handler = FLProtocolHandler(
            self.global_model,
            self.mobius_handler,
            self.result_manager,
            config
        )
        
        # Flask 앱
        self.app = Flask(__name__)
        self._register_routes()
    
    def _register_routes(self) -> None:
        """엔드포인트 등록"""
        # Mobius 엔드포인트
        mobius_bp = create_mobius_routes(self)
        self.app.register_blueprint(mobius_bp)
        
        # API 엔드포인트
        api_bp = create_api_routes(self)
        self.app.register_blueprint(api_bp)
        
        print("All routes registered")
    
    
    def start_flask(self):
        print(f"Starting Flask server at {self.host}:{self.port}...")
        self.app.run(host=self.host, port=self.port)
        
    def start(self):
        flask_thread = threading.Thread(target=self.start_flask)
        flask_thread.daemon = True  # 메인 스레드가 종료되면 Flask 스레드도 종료
        flask_thread.start()
        
        time.sleep(1) 
        self.mobius_handler.create_aeWatcher()

        # 서버 실행 유지
        while True:
            try:
                time.sleep(1)  # 메인 루프에서 대기
            except KeyboardInterrupt:
                print("Shutting down FLServer...")
                break
        
class SimpleConfig:
    def __init__(self, config_dict):
        # Model
        self.num_classes = config_dict['model']['num_classes']
        self.selected_labels = config_dict['model']['selected_labels']
        self.epoch_per_round = config_dict['model']['epoch_per_round']
        self.batch_size = config_dict['model']['batch_size']
        self.input_shape = tuple(config_dict['model'].get('input_shape', [256, 256, 3]))
        
        # Training
        self.MIN_NUM_WORKERS = config_dict['training']['MIN_NUM_WORKERS']
        self.MAx_NUM_ROUNDS = config_dict['training']['MAx_NUM_ROUNDS']
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = config_dict['training']['NUM_CLIENTS_CONTACTED_PER_ROUND']
        self.ROUNDS_BETWEEN_VALIDATIONS = config_dict['training']['ROUNDS_BETWEEN_VALIDATIONS']
        self.WINDOW_SIZE = config_dict['training']['WINDOW_SIZE']
        
        # Server
        self.host = config_dict['server']['host']
        self.port = config_dict['server']['port']
        
        # Mobius
        self.mobius_url = config_dict['mobius']['url']
        self.mobius_headers = config_dict['mobius']['headers']

        self.kd_on = config_dict['KD']
   
def load_config(config_path="config.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    if config_dict['model']['num_classes'] is None:
        raise ValueError(
            "num_classes is not set in config.json\n"
            "Please run preprocessing script first"
        )
    
    if not config_dict['model']['selected_labels']:
        raise ValueError(
            "selected_labels is empty in config.json\n"
            "Please run preprocessing script first"
        )
    
    return SimpleConfig(config_dict)   
   
def main():    
    config = load_config("config.json")
    server = FLServer(
        global_model=DeepLabV3PlusMobileNet,
        config=config
    )
    
    print(f"\nFL Server listening on server:5011")
    server.start()

if __name__ == '__main__':
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
    finally:
        elapsed = time.time() - start_time
        print(f"Total runtime: {elapsed:.2f} seconds")