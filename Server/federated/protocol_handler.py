"""
Federated Learning 프로토콜 핸들러
"""
import random
import json
import uuid
from typing import Set, List, Dict
from utils.pickle_utils import obj_to_pickle_string, pickle_string_to_obj
import sys
import time

class FLProtocolHandler:
    """연합학습 프로토콜 로직"""
    
    def __init__(self, global_model, mobius_handler, config):
        self.global_model = global_model
        self.mobius = mobius_handler
        self.config = config
        
        # 클라이언트 관리
        self.ready_client_sids: Set[str] = set()
        
        # 라운드 관리
        self.current_round = 0
        self.current_round_client_updates: List[Dict] = []
        self.eval_client_updates: List[Dict] = []
        
        # 모델 ID
        self.model_id = str(uuid.uuid4())
        
    def on_message(self, msg, client_id):
        payload = json.dumps(msg);
        
        try:
            # JSON 처리 시도
            payload = json.loads(payload)

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
                    'num_classes': self.config.num_classes,
                    'selected_labels': self.config.selected_labels,

                    #'data_split': (0.6, 0.3, 0.1), # train, test, valid
                    'epoch_per_round': 1,
                    'batch_size': 2
                }
            }
            self.mobius.publish(client_id+'FromS', data)
                        
        elif event == 'client_ready':
            data = payload['payload']
            print("client ready for training", client_id, data)
            self.ready_client_sids.add(client_id)
            
            #################################################
            ############ Delete if occurs error #############
            #################################################
            
            data={
                'event': 'global_update',
                'payload': {
                    'num_classes': self.config.num_classes,
                    'selected_labels': self.config.selected_labels,
                    'model_json': self.global_model.model.to_json(),
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                }
            }
            self.mobius.publish(client_id+'FromS', data)
            
            #################################################
            ############ Delete if occurs error #############
            #################################################

            if len(self.ready_client_sids) >= self.config.MIN_NUM_WORKERS and self.current_round == 0:
                self.train_next_round()
                
        elif event == 'client_update':
            data = payload['payload']
            print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", client_id)
            print('\033[1;35;0m handle client_update \033[0m')
            if data['round_number'] == self.current_round:
                self.current_round_client_updates.append(data)

                if len(self.current_round_client_updates) >= self.config.NUM_CLIENTS_CONTACTED_PER_ROUND:
                    # All selected clients for this round have replied.
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )

                    aggr_train_loss = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )

                    print("Aggregated training loss:", aggr_train_loss)

                    # New convergence checking logic:
                    if len(self.global_model.train_losses) >= (2 * self.config.WINDOW_SIZE):
                        # Get the last WINDOW_SIZE losses and the previous WINDOW_SIZE losses.
                        last_window = [entry[2] for entry in self.global_model.train_losses[-self.config.WINDOW_SIZE:]]
                        prev_window = [entry[2] for entry in self.global_model.train_losses[-(2 * self.config.WINDOW_SIZE):-self.config.WINDOW_SIZE]]
                        avg_recent = sum(last_window) / self.config.WINDOW_SIZE
                        avg_previous = sum(prev_window) / self.config.WINDOW_SIZE
                        print("Average loss for last", self.config.WINDOW_SIZE, "rounds:", avg_recent)
                        print("Average loss for previous", self.config.WINDOW_SIZE, "rounds:", avg_previous)
                        if avg_recent >= avg_previous:
                            print("Convergence criterion met (recent average loss is not lower than previous average). Triggering evaluation.")
                            self.stop_and_eval()
                            return

                    # if self.current_round >= FLServer.MAx_NUM_ROUNDS:
                    #     print("Maximum rounds reached. Triggering evaluation.")
                    #     self.stop_and_eval()
                    # else:
                    #     self.train_next_round()
                    
                    if self.current_round == self.config.MAx_NUM_ROUNDS:
                        print("Maximum rounds reached. Triggering evaluation.")
                        self.stop_and_eval()
                    else:
                        self.train_next_round()

                    self.current_round_client_updates = []
                    
        elif event == 'client_eval':
            data = payload['payload']
            if self.eval_client_updates is None:
                return
            print("handle client_eval", client_id)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            print('\033[1;35;0m == done == \033[0m')  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')
            # If the response contains a 'round_number', then it is an intermediate evaluation.
            if 'round_number' in data:
                round_number = data['round_number']
                print("Performance over the testing data set after round {}:".format(round_number))
            else:
                # Otherwise, training is complete. Print total training time cost.
                total_training_time = time.time() - self.global_model.training_start_time
                print('Total training time cost:', total_training_time)
      
            self.eval_client_updates = None  # Prevent further evaluation

    # Note: we assume that during training thlen(e #workers will be >= MI)N_NUM_WORKERS
    def train_next_round(self):
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        print("### Round ", self.current_round, "###")
        
        #################################################
        ############ Delete if occurs error #############
        #################################################
        
        for rid in list(self.ready_client_sids):
            data = {
                'event': 'global_update',
                'payload': {
                    'num_classes': self.config.num_classes,
                    'selected_labels': self.config.selected_labels,
                    'model_json': self.global_model.model.to_json(),
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                }
            }
            self.mobius.publish(rid+'FromS', data)
        print("Broadcasted global update to all ready clients.")
        
        client_sids_selected = random.sample(list(self.ready_client_sids), self.config.NUM_CLIENTS_CONTACTED_PER_ROUND)#为了提取出N个不同元素的样本用来(所有内容，需要的数量)
        print("request updates from", client_sids_selected)

        #################################################
        ############ Delete if occurs error #############
        #################################################


        # by default each client cnn is in its own "room"
        for rid in client_sids_selected:
            data = {
                'event': 'request_update', 
                'payload' : {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),

                    'weights_format': 'pickle',
                    'run_validation': self.current_round % self.config.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }
            }
            self.mobius.publish(rid+'FromS', data)
            
        #################################################
        ############ Delete if occurs error #############
        #################################################

        if self.current_round % self.config.ROUNDS_BETWEEN_VALIDATIONS == 0:
            print("Round {} is a validation round; requesting evaluation from all clients.".format(self.current_round))
            self.request_eval()

        #################################################
        ############ Delete if occurs error #############
        #################################################

    def stop_and_eval(self):
        #self.global_model.save("global_model.h5")
        self.eval_client_updates = []
        # self.stop_training = True  # 종료 플래그 설정
        for rid in self.ready_client_sids:
            data = {
                'event': 'stop_and_eval',
                'payload':  {
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle'
                }
            }
            self.mobius.publish(rid+'FromS', data)

    def request_eval(self):
        for rid in self.ready_client_sids:
            data = {
                'event': 'request_eval',
                'payload': {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle'
                }
            }
            self.mobius.publish(rid+'FromS', data)
