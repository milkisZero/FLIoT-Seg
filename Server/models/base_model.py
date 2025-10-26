import numpy as np
import time
import json
from utils.pickle_utils import obj_to_pickle_string, pickle_string_to_obj

class BaseGlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        
        # for convergence check
        self.prev_train_loss = None

        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.training_start_time = int(round(time.time())) 
        
    @staticmethod
    def fedavg(client_weights, client_sizes, current_weights):
        """
        Federated Averaging (FedAvg)
        w_global = Σ(n_k / n_total) * w_k
        """
        new_weights = [np.zeros_like(w, dtype=np.float32) for w in current_weights]
        total_size = np.sum(client_sizes)
        
        for c in range(len(client_weights)):
            # pickle string 역직렬화
            if isinstance(client_weights[c], str):
                client_weight_obj = pickle_string_to_obj(client_weights[c])
            else:
                client_weight_obj = client_weights[c]
            
            # 가중치 비율 계산
            weight_factor = client_sizes[c] / total_size
            
            # 각 레이어별 가중 평균
            for i in range(len(new_weights)):
                w_client = client_weight_obj[i]
                
                # numpy array 변환
                if not isinstance(w_client, np.ndarray):
                    w_client = np.array(w_client, dtype=np.float32)
                elif w_client.dtype != np.float32:
                    w_client = w_client.astype(np.float32)
                
                # 가중 평균 누적
                new_weights[i] += w_client * weight_factor
        return new_weights
        
    def update_weights(self, client_weights, client_sizes):
        self.current_weights = self.fedavg(
            client_weights,
            client_sizes,
            self.current_weights
        )

    def aggregate_loss_accuracy(self, client_losses, client_sizes):
        total_size = np.sum(client_sizes)
        print("-----client sizes-------", client_sizes)
        print('\033[1;35;0m client_losses \033[0m', client_losses)
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                           for i in range(len(client_sizes)))
        return aggr_loss

    def aggregate_train_loss_accuracy(self, client_losses, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss = self.aggregate_loss_accuracy(client_losses, client_sizes)
        self.train_losses.append([cur_round, cur_time, aggr_loss])
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