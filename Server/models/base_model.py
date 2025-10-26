import numpy as np
import time
import json
from utils.pickle_utils import obj_to_pickle_string, pickle_string_to_obj

class BaseGlobalModel(object):#类文档字符串
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
    def update_weights(self, client_weights, client_sizes):
        """
        Update current_weights using a weighted average of client_weights.
        """
        new_weights = [np.zeros_like(w, dtype=np.float32) for w in self.current_weights]
        total_size = np.sum(client_sizes)

        for c in range(len(client_weights)):
            if isinstance(client_weights[c], str):
                client_weight_obj = pickle_string_to_obj(client_weights[c])
            else:
                client_weight_obj = client_weights[c]
            for i in range(len(new_weights)):
                w_client = client_weight_obj[i]
                if not isinstance(w_client, np.ndarray):
                    w_client = np.array(w_client, dtype=np.float32)
                elif w_client.dtype != np.float32:
                    w_client = w_client.astype(np.float32)
                new_weights[i] += w_client * client_sizes[c] / total_size

        self.current_weights = new_weights

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