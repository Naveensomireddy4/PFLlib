# import time
# from flcore.clients.clientavg import clientAVG
# from flcore.servers.serverbase import Server
# import random
# import numpy as np

# class FedAvg(Server):
#     def __init__(self, args, times):
#         super().__init__(args, times)
#         self.temp = 1
#         self.temp_dropout_ratio = 0

#         # Initialize empty dictionaries to track times
#         self.client_learning_times = {}
#         self.client_upload_times = {}

#     def select_clients(self):
#         """Select a fraction of clients based on join_ratio and dropout_rate."""
#         print(self.num_clients,"............................")
#         num_selected_clients = max(1, int(self.join_ratio * self.num_clients))
#         print(len(self.clients),"............................")
#         # Ensure num_selected_clients does not exceed the actual number of clients
#         num_selected_clients = min(num_selected_clients, len(self.clients))
#         print(num_selected_clients,"............................")
       
#         # Handle cases where there are no clients or fewer clients than num_selected_clients
#         if num_selected_clients <= 0:
#             # Handle gracefully, for example, select all clients or raise an error
#             raise ValueError("Insufficient clients to select from.")
        
#         available_clients = random.sample(self.clients, num_selected_clients)

#         selected_clients = []
#         temp_selected_clients = []
#         dropped_clients = []

#         # Simulate dropout rate logic (you can adjust as needed)
#         dropout_rate = 0.2
#         temp_selected_clients = available_clients

#         threshold_val = dropout_rate * self.temp * 20
#         self.temp_dropout_ratio = dropout_rate * self.temp

#         a = 0
#         for client in temp_selected_clients:
#             a += 1
#             if a > threshold_val:
#                 selected_clients.append(client)
#             else:
#                 dropped_clients.append(client)

#         if len(selected_clients) == 0:
#             selected_clients = available_clients
#             dropped_clients = []
#             self.temp_dropout_ratio = 0

#         # Print dropped clients for debugging
#         print("Dropped clients:", [client.id for client in dropped_clients])

#         # Sort clients for consistent processing order
#         selected_clients.sort(key=lambda client: client.id)
#         dropped_clients.sort(key=lambda client: client.id)
#         temp_selected_clients.sort(key=lambda client: client.id)

#         return selected_clients, dropped_clients, temp_selected_clients

#     def euclidean_distance(self, weights1, weights2):
#         """Calculate the Euclidean distance between two sets of weights."""
#         w1 = np.concatenate([v.flatten() for v in weights1.values()])
#         w2 = np.concatenate([v.flatten() for v in weights2.values()])
#         return np.linalg.norm(w1 - w2)

#     def cosine_similarity(self, vec1, vec2):
#         """Calculate the cosine similarity between two vectors."""
#         vec1_array = np.concatenate([v.flatten() for v in vec1.values()])
#         vec2_array = np.concatenate([v.flatten() for v in vec2.values()])
#         dot_product = np.dot(vec1_array, vec2_array)
#         norm_vec1 = np.linalg.norm(vec1_array)
#         norm_vec2 = np.linalg.norm(vec2_array)
#         return dot_product / (norm_vec1 * norm_vec2)

#     def find_nearest_trained_client(self, dropped_client, trained_clients):
#         """Find the nearest trained client to a dropped client based on cosine similarity."""
#         max_similarity = -1
#         nearest_client = None
#         dropped_client_weights = dropped_client.get_weights()

#         for client in trained_clients:
#             trained_client_weights = client.get_weights()
#             similarity = self.cosine_similarity(dropped_client_weights, trained_client_weights)
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 nearest_client = client

#         return nearest_client

#     def finding_nearest_clients_to_dropped_client(self, dropped_clients):
#         """Find the nearest trained clients for each dropped client."""
#         dropped_clients_nearest_clients = []
#         for dropped_client in dropped_clients:
#             nearest_client = self.find_nearest_trained_client(dropped_client, self.selected_clients)
#             if nearest_client:
#                 dropped_clients_nearest_clients.append(nearest_client)
#         return dropped_clients_nearest_clients

#     def track_client_learning_time(self, client, time_cost):
#         """Track the learning time for a client."""
#         client_id = client.id
#         if client_id not in self.client_learning_times:
#             self.client_learning_times[client_id] = []
#         self.client_learning_times[client_id].append(time_cost)

#     def track_client_upload_time(self, client, time_cost):
#         """Track the upload time for a client."""
#         client_id = client.id
#         if client_id not in self.client_upload_times:
#             self.client_upload_times[client_id] = []
#         self.client_upload_times[client_id].append(time_cost)

#     def classify_clients(self):
#         """Classify clients into two batches based on their learning and upload speeds."""
#         learning_speed = [(client, np.mean(self.client_learning_times.get(client.id, [0]))) for client in self.clients]
#         upload_speed = [(client, np.mean(self.client_upload_times.get(client.id, [0]))) for client in self.clients]

#         # Sort clients based on average learning time
#         learning_speed.sort(key=lambda x: x[1])

#         # Sort clients based on average upload time
#         upload_speed.sort(key=lambda x: x[1])

#         # Extract slow learners and slow uploaders
#         slow_learners = [x[0] for x in learning_speed[:len(learning_speed)//2]]
#         slow_uploaders = [x[0] for x in upload_speed[:len(upload_speed)//2]]

#         # Create batches: Batch 1 contains slow learners and uploaders, Batch 2 contains others
#         self.batch1 = list(set(slow_learners + slow_uploaders))
#         self.batch2 = [client for client in self.clients if client not in self.batch1]

#         print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
#         print(f"Batch 2 clients: {[client.id for client in self.batch2]}")

#     def calculate_average_time_per_round(self, batch_clients):
#         """Calculate the average time per round for a batch of clients."""
#         total_learning_time = sum(np.mean(self.client_learning_times.get(client.id, [0])) for client in batch_clients)
#         total_upload_time = sum(np.mean(self.client_upload_times.get(client.id, [0])) for client in batch_clients)
#         avg_learning_time = total_learning_time / len(batch_clients)
#         avg_upload_time = total_upload_time / len(batch_clients)
#         return avg_learning_time + avg_upload_time

#     def adjust_rounds_for_batches(self):
#         """Adjust the number of rounds for each batch to optimize total training time."""
#         avg_time_batch1 = self.calculate_average_time_per_round(self.batch1)
#         avg_time_batch2 = self.calculate_average_time_per_round(self.batch2)

#         total_rounds = self.global_rounds - 5  # Remaining rounds after initial 5 rounds
#         total_time = (avg_time_batch1 + avg_time_batch2) / 2 * total_rounds

#         rounds_batch1 = int(total_time / avg_time_batch1)
#         rounds_batch2 = int(total_time / avg_time_batch2)

#         return rounds_batch1, rounds_batch2

#     def train_batch(self, batch_clients, num_rounds):
#         """Train a specific batch of clients for a given number of rounds."""
#         for round_num in range(num_rounds):
#             print(f"Training round {round_num+1}/{num_rounds} for batch {batch_clients[0].id}")
#             selected_clients, dropped_clients, temp_selected_clients = self.select_clients()
#             self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients)
#             self.send_models()

#             cnt = 1
#             for client in batch_clients:
#                 st = time.time()
#                 client.train()
#                 train_accuracy = client.get_train_accuracy()
#                 learning_time = time.time() - st
#                 self.track_client_learning_time(client, learning_time)
#                 print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
#                 cnt += 1

#             for i in range(len(dropped_clients)):
#                 dropped_clients[i].set_weights(self.dropped_clients_nearest_clients[i].get_weights())

#             st = time.time()
#             self.receive_models(dropout_ratio=0, required_clients=batch_clients)
#             upload_time = time.time() - st
#             for client in selected_clients:
#                 self.track_client_upload_time(client, upload_time / len(selected_clients))
#             self.aggregate_parameters()

#     def train(self):
#         """Train the federated averaging model."""
#         for i in range(self.global_rounds + 1):
#             s_t = time.time()

#             if i < 1:
#                 # Initial round: Train all clients
#                 self.selected_clients, dropped_clients, self.temp_selected_clients = self.select_clients()
#                 self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients)
#                 self.send_models()

#                 if i % self.eval_gap == 0:
#                     print(f"\n-------------Round number: {i}-------------")
#                     print("\nEvaluate global model")
#                     self.evaluate()

#                 print("clients:", len(self.selected_clients))

#                 cnt = 1
#                 for client in self.selected_clients:
#                     st = time.time()
#                     client.train()
#                     train_accuracy = client.get_train_accuracy()
#                     learning_time = time.time() - st
#                     self.track_client_learning_time(client, learning_time)
#                     print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
#                     cnt += 1

#                 for i in range(len(dropped_clients)):
#                     dropped_clients[i].set_weights(self.dropped_clients_nearest_clients[i].get_weights())

#                 st = time.time()
#                 self.receive_models(dropout_ratio=0, required_clients=self.selected_clients)
#                 upload_time = time.time() - st
#                 for client in self.selected_clients:
#                     self.track_client_upload_time(client, upload_time / len(self.selected_clients))
#                 self.aggregate_parameters()
#             else:
#                 # After initial rounds: Classify clients into batches and train each batch
#                 if i == 1:
#                     self.classify_clients()
#                     self.rounds_batch1, self.rounds_batch2 = self.adjust_rounds_for_batches()

#                 # Train batch 1
#                 print(f"\n-------------Training Batch 1-------------")
#                 self.train_batch(self.batch1, self.rounds_batch1)

#                 # Train batch 2
#                 print(f"\n-------------Training Batch 2-------------")
#                 self.train_batch(self.batch2, self.rounds_batch2)

#                 break  # Remove this if you want to continue training beyond 2 batches

#             if i % self.eval_gap == 0:
#                 print(f"\n-------------Round number: {i}-------------")
#                 print("\nEvaluate global model")
#                 self.evaluate()

#             self.Budget.append(time.time() - s_t)
#             print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

#             if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
#                 break

#         print("\nBest accuracy.")
#         print(max(self.rs_test_acc))

#         if len(self.Budget) > 1:
#             print("\nAverage time cost per round.")
#             print(sum(self.Budget[1:]) / len(self.Budget[1:]))
#         else:
#             print("\nAverage time cost per round: Insufficient data.")


# import time
# import random
# import numpy as np
# from flcore.servers.serverbase import Server
# from flcore.clients.clientavg import clientAVG
# import threading
# import wandb

# wandb.init(project='rounds', entity='naveen2112')

# class FedAvg(Server):
#     def __init__(self, args, times):
#         super().__init__(args, times)
#         self.temp = 1
#         self.temp_droupout_ratio = 0

#         self.set_slow_clients()  # Assuming this method exists
#         self.set_clients(clientAVG)  # Assuming this method sets up clients
#         print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
#         print("Finished creating server and clients.")

#         self.Budget = []
#         self.client_learning_times = {}
#         self.client_upload_times = {}
#         self.restrict_k=5;

#     def select_clients(self):
#         """Select a fraction of clients based on join_ratio and dropout_rate."""
#         num_selected_clients = max(1, int(self.join_ratio * self.num_clients))
#         available_clients = random.sample(self.clients, num_selected_clients)

#         selected_clients = []
#         temp_selected_clients = []
#         dropped_clients = []
#         dropout_rate = 0.5  # Example dropout rate, adjust as needed

#         temp_selected_clients = available_clients

#         threshold_val = dropout_rate * self.temp * 20
#         self.temp_droupout_ratio = dropout_rate * self.temp

#         a = 0
#         for client in temp_selected_clients:
#             a += 1
#             if a > threshold_val:
#                 selected_clients.append(client)
#             else:
#                 dropped_clients.append(client)

#         if len(selected_clients) == 0:
#             selected_clients = available_clients
#             dropped_clients = []
#             self.temp_droupout_ratio = 0

#         print("Dropped clients:", [client.id for client in dropped_clients])

#         selected_clients.sort(key=lambda client: client.id)
#         dropped_clients.sort(key=lambda client: client.id)
#         temp_selected_clients.sort(key=lambda client: client.id)

#         return selected_clients, dropped_clients, temp_selected_clients

#     def euclidean_distance(self, weights1, weights2):
#         """Calculate the Euclidean distance between two sets of weights."""
#         w1 = np.concatenate([v.flatten() for v in weights1.values()])
#         w2 = np.concatenate([v.flatten() for v in weights2.values()])
#         return np.linalg.norm(w1 - w2)

#     def ordered_dict_to_array(self, ordered_dict):
#         """Convert OrderedDict of weights to numpy array."""
#         return np.concatenate([value.flatten() for value in ordered_dict.values()])

#     def cosine_similarity(self, vec1, vec2):
#         """Calculate the cosine similarity between two weight vectors."""
#         vec1_array = self.ordered_dict_to_array(vec1)
#         vec2_array = self.ordered_dict_to_array(vec2)
#         dot_product = np.dot(vec1_array, vec2_array)
#         norm_vec1 = np.linalg.norm(vec1_array)
#         norm_vec2 = np.linalg.norm(vec2_array)
#         return dot_product / (norm_vec1 * norm_vec2)

#     def find_nearest_trained_client(self, dropped_client, trained_clients):
#         """Find the nearest trained client to a dropped client based on cosine similarity."""
#         max_similarity = -1000
#         nearest_client = None
#         temp_c = None
#         dropped_client_weights = dropped_client.get_weights()  # Assuming this method exists

#         for client in trained_clients:
#             trained_client_weights = client.get_weights()  # Assuming this method exists
#             similarity = self.cosine_similarity(dropped_client_weights, trained_client_weights)
#             temp_c = client
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 nearest_client = client
#         if nearest_client is None:
#             nearest_client = temp_c
#         return nearest_client

#     def finding_nearest_clients_to_dropped_client(self, dropped_clients,rem_client):
#         """Find the nearest trained clients for each dropped client."""
#         dropped_clients_nearest_clients = []
#         for dropped_client in dropped_clients:
#             nearest_client = self.find_nearest_trained_client(dropped_client, rem_client)
#             if nearest_client:
#                 dropped_clients_nearest_clients.append(nearest_client)
#         return dropped_clients_nearest_clients

#     def track_client_learning_time(self, client, learning_time):
#         """Track the learning time of a client."""
#         if client.id in self.client_learning_times:
#             self.client_learning_times[client.id].append(learning_time)
#         else:
#             self.client_learning_times[client.id] = [learning_time]

#     def track_client_upload_time(self, client, upload_time):
#         """Track the upload time of a client."""
#         if client.id in self.client_upload_times:
#             self.client_upload_times[client.id].append(upload_time)
#         else:
#             self.client_upload_times[client.id] = [upload_time]

#     def classify_clients(self):
#         """Classify clients into two batches based on their learning and upload speeds."""
#         learning_speed = [(client, np.mean(self.client_learning_times.get(client.id, [0]))) for client in self.clients]
#         upload_speed = [(client, np.mean(self.client_upload_times.get(client.id, [0]))) for client in self.clients]

#         # Sort clients based on average learning time
#         learning_speed.sort(key=lambda x: x[1])

#         # Sort clients based on average upload time
#         upload_speed.sort(key=lambda x: x[1])

#         # Extract slow learners and slow uploaders
#         slow_learners = [x[0] for x in learning_speed[:len(learning_speed) // 2]]
#         slow_uploaders = [x[0] for x in upload_speed[:len(upload_speed) // 2]]

#         # Create batches: Batch 1 contains slow learners and uploaders, Batch 2 contains others
#         self.batch1 = list(set(slow_learners + slow_uploaders))
#         self.batch2 = [client for client in self.clients if client not in self.batch1]

#         print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
#         print(f"Batch 2 clients: {[client.id for client in self.batch2]}")

#     def calculate_average_time_per_round(self, batch_clients):
#         """Calculate the average time per round for a batch of clients."""
#         total_learning_time = sum(np.mean(self.client_learning_times.get(client.id, [0])) for client in batch_clients)
#         total_upload_time = sum(np.mean(self.client_upload_times.get(client.id, [0])) for client in batch_clients)
#         avg_learning_time = total_learning_time / len(batch_clients)
#         avg_upload_time = total_upload_time / len(batch_clients)
#         return avg_learning_time + avg_upload_time

#     def adjust_rounds_for_batches(self):
#         """Adjust the number of rounds for each batch to optimize total training time."""
#         print("hello")
#         avg_time_batch1 = self.calculate_average_time_per_round(self.batch1)
#         avg_time_batch2 = self.calculate_average_time_per_round(self.batch2)

#         total_rounds = self.global_rounds - self.restrict_k         # Remaining rounds after initial 5 rounds
#         total_time = (avg_time_batch1 + avg_time_batch2) / 2 * total_rounds

#         rounds_batch1 = int(total_time / avg_time_batch1)
#         rounds_batch2 = int(total_time / avg_time_batch2)

#         return rounds_batch1, rounds_batch2
#     def train_batch_threaded(self, batch, rounds):
#         """Thread function to train a batch for specified rounds."""
#         for _ in range(rounds):
#             # Example logic, replace with your actual training process
#             time.sleep(1)  # Simulate training time
#             print(f"Training {batch} - round {_+1}")


    
#     def train_client(self, client):
#         st = time.time()
#         client.train()  # Replace with actual training logic
#         train_accuracy = client.get_train_accuracy()  # Assuming this method exists
#         learning_time = time.time() - st
#         self.track_client_learning_time(client, learning_time)
#         print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
        

#     def train_batch(self, batch_clients, num_rounds):
#         """Train a specific batch of clients for a given number of rounds."""
#         for round_num in range(num_rounds):
#             print(f"Training round {round_num+1}/{num_rounds} for batch {batch_clients[0].id}")
#             #selected_clients, dropped_clients, temp_selected_clients = self.select_clients()
#             # self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients,selected_clients)
#             # batch_clients = random.shuffle(batch_clients)
#             total_clients = random.sample(batch_clients,len(batch_clients))
#             print(type(batch_clients))
#             dr= 0.5
#             drop_len = dr*len(batch_clients)
#             print("."*25,drop_len,"."*25)
#             friend_client=[]
#             drop_client=[]
#             a=1;
#             for client in total_clients:
#               if(a>drop_len):
#                    friend_client.append(client)
#               else:
#                   drop_client.append(client)
#               a=a+1    

#             drop_client_friend_client =self.finding_nearest_clients_to_dropped_client(drop_client,friend_client)      

#             self.send_models()#set weights

#             threads = []
#             for client in friend_client:
#                 thread = threading.Thread(target=self.train_client, args=(client,))
#                 threads.append(thread)
#                 thread.start()

#             for thread in threads:
#                 thread.join()
#             print("."*25, "leng of drop and ffrnd")
#             print(len(drop_client))
#             print(len(drop_client_friend_client) )
#             for i in range(min(len(drop_client),len(drop_client_friend_client))):
#                 drop_client[i].set_weights(drop_client_friend_client[i].get_weights())

#             st = time.time()
#             self.receive_models(dropout_ratio=0, required_clients=self.clients)
#             upload_time = time.time() - st
#             for client in self.clients:
#                 self.track_client_upload_time(client, upload_time / len(self.clients))
#             self.aggregate_parameters()

#             #self.send_models()
#             # print("\nEvaluate global model")
#             # self.evaluate()

#     def train(self):
#         """Train the federated averaging model."""
#         for i in range(self.global_rounds + 1):
#             s_t = time.time()

#             if i <self.restrict_k :
#                 # Initial round: Train all clients
#                 self.selected_clients, dropped_clients, self.temp_selected_clients = self.select_clients()
#                 self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients,self.selected_clients)
#                 print("drop classif com")
#                 self.send_models()

#                 if i % self.eval_gap == 0:
#                     print(f"\n-------------Round number: {i}-------------")
#                     print("\nEvaluate global model")
#                     self.evaluate()

#                 print("clients:", len(self.selected_clients))

#                 cnt = 1
#                 for client in self.selected_clients:
#                     st = time.time()
#                     client.train()
#                     train_accuracy = client.get_train_accuracy()
#                     learning_time = time.time() - st
#                     self.track_client_learning_time(client, learning_time)
#                     print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
#                     cnt += 1

#                 for i in range(len(dropped_clients)):
#                     dropped_clients[i].set_weights(self.dropped_clients_nearest_clients[i].get_weights())

#                 st = time.time()
#                 self.receive_models(dropout_ratio=0, required_clients=self.temp_selected_clients)
#                 upload_time = time.time() - st

#                 for client in self.temp_selected_clients:
#                     self.track_client_upload_time(client, upload_time / len(self.temp_selected_clients))
#                 self.aggregate_parameters()

#             else:
#                 # Batch training after the initial round
#                 self.classify_clients()
#                 rounds_batch1, rounds_batch2 = self.adjust_rounds_for_batches()
#                 # rounds_batch1 = min(rounds_batch1,2*rounds_batch2)
                 
#                 # if rounds_batch2 > 0:
#                 #     print(f"\n-------------Batch 2 Training: Round {i}-------------")
#                 #     self.train_batch(self.batch2, rounds_batch2)
                                  
#                 # if rounds_batch1 > 0:
#                 #     print(f"\n-------------Batch 1 Training: Round {i}-------------")
#                 #     self.train_batch(self.batch1, rounds_batch1)
#                 #  """Train both batch1 and batch2 concurrently using threads."""
#                 # rounds_batch1, rounds_batch2 = self.adjust_rounds_for_batches()

#                 # Create threads for training batch1 and batch2
#                 print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
#                 print(f"Batch 2 clients: {[client.id for client in self.batch2]}")
#                 thread_batch1 = threading.Thread(target=self.train_batch, args=(self.batch1, rounds_batch1))
#                 thread_batch2 = threading.Thread(target=self.train_batch, args=(self.batch2, rounds_batch2))


#                 # Start both threads
#                 thread_batch1.start()
#                 thread_batch2.start()

#                 # Wait for both threads to complete
#                 thread_batch1.join()
#                 thread_batch2.join()
                 

                

#                 # Aggregate global model after batch training
#                 self.send_models()
                
#                 print("\nEvaluate global model")
#                 self.evaluate()

#                 st = time.time()
#                 self.receive_models(dropout_ratio=self.temp_droupout_ratio, required_clients=self.temp_selected_clients)
#                 upload_time = time.time() - st

#                 for client in self.temp_selected_clients:
#                     self.track_client_upload_time(client, upload_time / len(self.temp_selected_clients))
#                 self.aggregate_parameters()

#                 e_t = time.time()
#                 print(f"Round time cost: {e_t - s_t}")
#                 break   
# import time
# from flcore.clients.clientavg import clientAVG
# from flcore.servers.serverbase import Server
# import random
# import numpy as np

# class FedAvg(Server):
#     def __init__(self, args, times):
#         super().__init__(args, times)
#         self.temp = 1
#         self.temp_dropout_ratio = 0

#         # Initialize empty dictionaries to track times
#         self.client_learning_times = {}
#         self.client_upload_times = {}

#     def select_clients(self):
#         """Select a fraction of clients based on join_ratio and dropout_rate."""
#         print(self.num_clients,"............................")
#         num_selected_clients = max(1, int(self.join_ratio * self.num_clients))
#         print(len(self.clients),"............................")
#         # Ensure num_selected_clients does not exceed the actual number of clients
#         num_selected_clients = min(num_selected_clients, len(self.clients))
#         print(num_selected_clients,"............................")
       
#         # Handle cases where there are no clients or fewer clients than num_selected_clients
#         if num_selected_clients <= 0:
#             # Handle gracefully, for example, select all clients or raise an error
#             raise ValueError("Insufficient clients to select from.")
        
#         available_clients = random.sample(self.clients, num_selected_clients)

#         selected_clients = []
#         temp_selected_clients = []
#         dropped_clients = []

#         # Simulate dropout rate logic (you can adjust as needed)
#         dropout_rate = 0.2
#         temp_selected_clients = available_clients

#         threshold_val = dropout_rate * self.temp * 20
#         self.temp_dropout_ratio = dropout_rate * self.temp

#         a = 0
#         for client in temp_selected_clients:
#             a += 1
#             if a > threshold_val:
#                 selected_clients.append(client)
#             else:
#                 dropped_clients.append(client)

#         if len(selected_clients) == 0:
#             selected_clients = available_clients
#             dropped_clients = []
#             self.temp_dropout_ratio = 0

#         # Print dropped clients for debugging
#         print("Dropped clients:", [client.id for client in dropped_clients])

#         # Sort clients for consistent processing order
#         selected_clients.sort(key=lambda client: client.id)
#         dropped_clients.sort(key=lambda client: client.id)
#         temp_selected_clients.sort(key=lambda client: client.id)

#         return selected_clients, dropped_clients, temp_selected_clients

#     def euclidean_distance(self, weights1, weights2):
#         """Calculate the Euclidean distance between two sets of weights."""
#         w1 = np.concatenate([v.flatten() for v in weights1.values()])
#         w2 = np.concatenate([v.flatten() for v in weights2.values()])
#         return np.linalg.norm(w1 - w2)

#     def cosine_similarity(self, vec1, vec2):
#         """Calculate the cosine similarity between two vectors."""
#         vec1_array = np.concatenate([v.flatten() for v in vec1.values()])
#         vec2_array = np.concatenate([v.flatten() for v in vec2.values()])
#         dot_product = np.dot(vec1_array, vec2_array)
#         norm_vec1 = np.linalg.norm(vec1_array)
#         norm_vec2 = np.linalg.norm(vec2_array)
#         return dot_product / (norm_vec1 * norm_vec2)

#     def find_nearest_trained_client(self, dropped_client, trained_clients):
#         """Find the nearest trained client to a dropped client based on cosine similarity."""
#         max_similarity = -1
#         nearest_client = None
#         dropped_client_weights = dropped_client.get_weights()

#         for client in trained_clients:
#             trained_client_weights = client.get_weights()
#             similarity = self.cosine_similarity(dropped_client_weights, trained_client_weights)
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 nearest_client = client

#         return nearest_client

#     def finding_nearest_clients_to_dropped_client(self, dropped_clients):
#         """Find the nearest trained clients for each dropped client."""
#         dropped_clients_nearest_clients = []
#         for dropped_client in dropped_clients:
#             nearest_client = self.find_nearest_trained_client(dropped_client, self.selected_clients)
#             if nearest_client:
#                 dropped_clients_nearest_clients.append(nearest_client)
#         return dropped_clients_nearest_clients

#     def track_client_learning_time(self, client, time_cost):
#         """Track the learning time for a client."""
#         client_id = client.id
#         if client_id not in self.client_learning_times:
#             self.client_learning_times[client_id] = []
#         self.client_learning_times[client_id].append(time_cost)

#     def track_client_upload_time(self, client, time_cost):
#         """Track the upload time for a client."""
#         client_id = client.id
#         if client_id not in self.client_upload_times:
#             self.client_upload_times[client_id] = []
#         self.client_upload_times[client_id].append(time_cost)

#     def classify_clients(self):
#         """Classify clients into two batches based on their learning and upload speeds."""
#         learning_speed = [(client, np.mean(self.client_learning_times.get(client.id, [0]))) for client in self.clients]
#         upload_speed = [(client, np.mean(self.client_upload_times.get(client.id, [0]))) for client in self.clients]

#         # Sort clients based on average learning time
#         learning_speed.sort(key=lambda x: x[1])

#         # Sort clients based on average upload time
#         upload_speed.sort(key=lambda x: x[1])

#         # Extract slow learners and slow uploaders
#         slow_learners = [x[0] for x in learning_speed[:len(learning_speed)//2]]
#         slow_uploaders = [x[0] for x in upload_speed[:len(upload_speed)//2]]

#         # Create batches: Batch 1 contains slow learners and uploaders, Batch 2 contains others
#         self.batch1 = list(set(slow_learners + slow_uploaders))
#         self.batch2 = [client for client in self.clients if client not in self.batch1]

#         print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
#         print(f"Batch 2 clients: {[client.id for client in self.batch2]}")

#     def calculate_average_time_per_round(self, batch_clients):
#         """Calculate the average time per round for a batch of clients."""
#         total_learning_time = sum(np.mean(self.client_learning_times.get(client.id, [0])) for client in batch_clients)
#         total_upload_time = sum(np.mean(self.client_upload_times.get(client.id, [0])) for client in batch_clients)
#         avg_learning_time = total_learning_time / len(batch_clients)
#         avg_upload_time = total_upload_time / len(batch_clients)
#         return avg_learning_time + avg_upload_time

#     def adjust_rounds_for_batches(self):
#         """Adjust the number of rounds for each batch to optimize total training time."""
#         avg_time_batch1 = self.calculate_average_time_per_round(self.batch1)
#         avg_time_batch2 = self.calculate_average_time_per_round(self.batch2)

#         total_rounds = self.global_rounds - 5  # Remaining rounds after initial 5 rounds
#         total_time = (avg_time_batch1 + avg_time_batch2) / 2 * total_rounds

#         rounds_batch1 = int(total_time / avg_time_batch1)
#         rounds_batch2 = int(total_time / avg_time_batch2)

#         return rounds_batch1, rounds_batch2

#     def train_batch(self, batch_clients, num_rounds):
#         """Train a specific batch of clients for a given number of rounds."""
#         for round_num in range(num_rounds):
#             print(f"Training round {round_num+1}/{num_rounds} for batch {batch_clients[0].id}")
#             selected_clients, dropped_clients, temp_selected_clients = self.select_clients()
#             self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients)
#             self.send_models()

#             cnt = 1
#             for client in batch_clients:
#                 st = time.time()
#                 client.train()
#                 train_accuracy = client.get_train_accuracy()
#                 learning_time = time.time() - st
#                 self.track_client_learning_time(client, learning_time)
#                 print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
#                 cnt += 1

#             for i in range(len(dropped_clients)):
#                 dropped_clients[i].set_weights(self.dropped_clients_nearest_clients[i].get_weights())

#             st = time.time()
#             self.receive_models(dropout_ratio=0, required_clients=batch_clients)
#             upload_time = time.time() - st
#             for client in selected_clients:
#                 self.track_client_upload_time(client, upload_time / len(selected_clients))
#             self.aggregate_parameters()

#     def train(self):
#         """Train the federated averaging model."""
#         for i in range(self.global_rounds + 1):
#             s_t = time.time()

#             if i < 1:
#                 # Initial round: Train all clients
#                 self.selected_clients, dropped_clients, self.temp_selected_clients = self.select_clients()
#                 self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients)
#                 self.send_models()

#                 if i % self.eval_gap == 0:
#                     print(f"\n-------------Round number: {i}-------------")
#                     print("\nEvaluate global model")
#                     self.evaluate()

#                 print("clients:", len(self.selected_clients))

#                 cnt = 1
#                 for client in self.selected_clients:
#                     st = time.time()
#                     client.train()
#                     train_accuracy = client.get_train_accuracy()
#                     learning_time = time.time() - st
#                     self.track_client_learning_time(client, learning_time)
#                     print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
#                     cnt += 1

#                 for i in range(len(dropped_clients)):
#                     dropped_clients[i].set_weights(self.dropped_clients_nearest_clients[i].get_weights())

#                 st = time.time()
#                 self.receive_models(dropout_ratio=0, required_clients=self.selected_clients)
#                 upload_time = time.time() - st
#                 for client in self.selected_clients:
#                     self.track_client_upload_time(client, upload_time / len(self.selected_clients))
#                 self.aggregate_parameters()
#             else:
#                 # After initial rounds: Classify clients into batches and train each batch
#                 if i == 1:
#                     self.classify_clients()
#                     self.rounds_batch1, self.rounds_batch2 = self.adjust_rounds_for_batches()

#                 # Train batch 1
#                 print(f"\n-------------Training Batch 1-------------")
#                 self.train_batch(self.batch1, self.rounds_batch1)

#                 # Train batch 2
#                 print(f"\n-------------Training Batch 2-------------")
#                 self.train_batch(self.batch2, self.rounds_batch2)

#                 break  # Remove this if you want to continue training beyond 2 batches

#             if i % self.eval_gap == 0:
#                 print(f"\n-------------Round number: {i}-------------")
#                 print("\nEvaluate global model")
#                 self.evaluate()

#             self.Budget.append(time.time() - s_t)
#             print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

#             if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
#                 break

#         print("\nBest accuracy.")
#         print(max(self.rs_test_acc))

#         if len(self.Budget) > 1:
#             print("\nAverage time cost per round.")
#             print(sum(self.Budget[1:]) / len(self.Budget[1:]))
#         else:
#             print("\nAverage time cost per round: Insufficient data.")


# import time
# import random
# import numpy as np
# from flcore.servers.serverbase import Server
# from flcore.clients.clientavg import clientAVG
# import threading
# import wandb

# wandb.init(project='rounds', entity='naveen2112')

# class FedAvg(Server):
#     def __init__(self, args, times):
#         super().__init__(args, times)
#         self.temp = 1
#         self.temp_droupout_ratio = 0

#         self.set_slow_clients()  # Assuming this method exists
#         self.set_clients(clientAVG)  # Assuming this method sets up clients
#         print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
#         print("Finished creating server and clients.")

#         self.Budget = []
#         self.client_learning_times = {}
#         self.client_upload_times = {}
#         self.restrict_k=20;
#         self.dropout_ratio_dr=0

#     def select_clients(self):
#         """Select a fraction of clients based on join_ratio and dropout_rate."""
#         num_selected_clients = max(1, int(self.join_ratio * self.num_clients))
#         available_clients = random.sample(self.clients, num_selected_clients)

#         selected_clients = []
#         temp_selected_clients = []
#         dropped_clients = []
#         dropout_rate =  self.dropout_ratio_dr  # Example dropout rate, adjust as needed

#         temp_selected_clients = available_clients

#         threshold_val = dropout_rate * self.temp * 20
#         self.temp_droupout_ratio = dropout_rate * self.temp

#         a = 0
#         for client in temp_selected_clients:
#             a += 1
#             if a > threshold_val:
#                 selected_clients.append(client)
#             else:
#                 dropped_clients.append(client)

#         if len(selected_clients) == 0:
#             selected_clients = available_clients
#             dropped_clients = []
#             self.temp_droupout_ratio = 0

#         print("Dropped clients:", [client.id for client in dropped_clients])

#         selected_clients.sort(key=lambda client: client.id)
#         dropped_clients.sort(key=lambda client: client.id)
#         temp_selected_clients.sort(key=lambda client: client.id)

#         return selected_clients, dropped_clients, temp_selected_clients

#     def euclidean_distance(self, weights1, weights2):
#         """Calculate the Euclidean distance between two sets of weights."""
#         w1 = np.concatenate([v.flatten() for v in weights1.values()])
#         w2 = np.concatenate([v.flatten() for v in weights2.values()])
#         return np.linalg.norm(w1 - w2)

#     def ordered_dict_to_array(self, ordered_dict):
#         """Convert OrderedDict of weights to numpy array."""
#         return np.concatenate([value.flatten() for value in ordered_dict.values()])

#     def cosine_similarity(self, vec1, vec2):
#         """Calculate the cosine similarity between two weight vectors."""
#         vec1_array = self.ordered_dict_to_array(vec1)
#         vec2_array = self.ordered_dict_to_array(vec2)
#         dot_product = np.dot(vec1_array, vec2_array)
#         norm_vec1 = np.linalg.norm(vec1_array)
#         norm_vec2 = np.linalg.norm(vec2_array)
#         return dot_product / (norm_vec1 * norm_vec2)

#     def find_nearest_trained_client(self, dropped_client, trained_clients):
#         """Find the nearest trained client to a dropped client based on cosine similarity."""
#         max_similarity = -1000
#         nearest_client = None
#         temp_c = None
#         dropped_client_weights = dropped_client.get_weights()  # Assuming this method exists

#         for client in trained_clients:
#             trained_client_weights = client.get_weights()  # Assuming this method exists
#             similarity = self.cosine_similarity(dropped_client_weights, trained_client_weights)
#             temp_c = client
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 nearest_client = client
#         if nearest_client is None:
#             nearest_client = temp_c
#         return nearest_client

#     def finding_nearest_clients_to_dropped_client(self, dropped_clients,rem_client):
#         """Find the nearest trained clients for each dropped client."""
#         dropped_clients_nearest_clients = []
#         for dropped_client in dropped_clients:
#             nearest_client = self.find_nearest_trained_client(dropped_client, rem_client)
#             if nearest_client:
#                 dropped_clients_nearest_clients.append(nearest_client)
#         return dropped_clients_nearest_clients

#     def track_client_learning_time(self, client, learning_time):
#         """Track the learning time of a client."""
#         if client.id in self.client_learning_times:
#             self.client_learning_times[client.id].append(learning_time)
#         else:
#             self.client_learning_times[client.id] = [learning_time]

#     def track_client_upload_time(self, client, upload_time):
#         """Track the upload time of a client."""
#         if client.id in self.client_upload_times:
#             self.client_upload_times[client.id].append(upload_time)
#         else:
#             self.client_upload_times[client.id] = [upload_time]

#     def classify_clients(self):
#         """Classify clients into two batches based on their learning and upload speeds."""
#         learning_speed = [(client, np.mean(self.client_learning_times.get(client.id, [0]))) for client in self.clients]
#         upload_speed = [(client, np.mean(self.client_upload_times.get(client.id, [0]))) for client in self.clients]

#         # Sort clients based on average learning time
#         learning_speed.sort(key=lambda x: x[1])

#         # Sort clients based on average upload time
#         upload_speed.sort(key=lambda x: x[1])

#         # Extract slow learners and slow uploaders
#         slow_learners = [x[0] for x in learning_speed[:len(learning_speed) // 2]]
#         slow_uploaders = [x[0] for x in upload_speed[:len(upload_speed) // 2]]

#         # Create batches: Batch 1 contains slow learners and uploaders, Batch 2 contains others
#         self.batch1 = list(set(slow_learners + slow_uploaders))
#         self.batch2 = [client for client in self.clients if client not in self.batch1]

#         print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
#         print(f"Batch 2 clients: {[client.id for client in self.batch2]}")

#     def calculate_average_time_per_round(self, batch_clients):
#         """Calculate the average time per round for a batch of clients."""
#         total_learning_time = sum(np.mean(self.client_learning_times.get(client.id, [0])) for client in batch_clients)
#         total_upload_time = sum(np.mean(self.client_upload_times.get(client.id, [0])) for client in batch_clients)
#         avg_learning_time = total_learning_time / len(batch_clients)
#         avg_upload_time = total_upload_time / len(batch_clients)
#         return avg_learning_time + avg_upload_time

#     def adjust_rounds_for_batches(self):
#         """Adjust the number of rounds for each batch to optimize total training time."""
#         print("hello")
#         avg_time_batch1 = self.calculate_average_time_per_round(self.batch1)
#         avg_time_batch2 = self.calculate_average_time_per_round(self.batch2)

#         total_rounds = self.global_rounds - self.restrict_k         # Remaining rounds after initial 5 rounds
#         total_time = (avg_time_batch1 + avg_time_batch2) / 2 * total_rounds

#         rounds_batch1 = int(total_time / avg_time_batch1)
#         rounds_batch2 = int(total_time / avg_time_batch2)

#         return rounds_batch1, rounds_batch2
#     def train_batch_threaded(self, batch, rounds):
#         """Thread function to train a batch for specified rounds."""
#         for _ in range(rounds):
#             # Example logic, replace with your actual training process
#             time.sleep(1)  # Simulate training time
#             print(f"Training {batch} - round {_+1}")


    
#     def train_client(self, client):
#         st = time.time()
#         client.train()  # Replace with actual training logic
#         train_accuracy = client.get_train_accuracy()  # Assuming this method exists
#         learning_time = time.time() - st
#         self.track_client_learning_time(client, learning_time)
#         print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
        

#     def train_batch(self, batch_clients, num_rounds):
#         """Train a specific batch of clients for a given number of rounds."""
#         for round_num in range(num_rounds):
#             print(f"Training round {round_num+1}/{num_rounds} for batch {batch_clients[0].id}")
#             #selected_clients, dropped_clients, temp_selected_clients = self.select_clients()
#             # self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients,selected_clients)
#             # batch_clients = random.shuffle(batch_clients)
#             total_clients = random.sample(batch_clients,len(batch_clients))
#             print(type(batch_clients))
#             dr= self.dropout_ratio_dr
#             drop_len = dr*len(batch_clients)
#             print("."*25,drop_len,"."*25)
#             friend_client=[]
#             drop_client=[]
#             a=1;
#             for client in total_clients:
#               if(a>drop_len):
#                    friend_client.append(client)
#               else:
#                   drop_client.append(client)
#               a=a+1    

#             drop_client_friend_client =self.finding_nearest_clients_to_dropped_client(drop_client,friend_client)      

#             self.send_models()#set weights

#             threads = []
#             for client in friend_client:
#                 thread = threading.Thread(target=self.train_client, args=(client,))
#                 threads.append(thread)
#                 thread.start()

#             for thread in threads:
#                 thread.join()
#             print("."*25, "leng of drop and ffrnd")
#             print(len(drop_client))
#             print(len(drop_client_friend_client) )
#             for i in range(min(len(drop_client),len(drop_client_friend_client))):
#                 drop_client[i].set_weights(drop_client_friend_client[i].get_weights())

#             st = time.time()
#             self.receive_models(dropout_ratio=0, required_clients=self.clients)
#             upload_time = time.time() - st
#             for client in self.clients:
#                 self.track_client_upload_time(client, upload_time / len(self.clients))
#             self.aggregate_parameters()

#             self.send_models()
#             print("\nEvaluate global model")
#             self.evaluate()

#     def train(self):
#         """Train the federated averaging model."""
#         for i in range(self.global_rounds + 1):
#             s_t = time.time()

#             if i <self.restrict_k :
#                 # Initial round: Train all clients
#                 self.selected_clients, dropped_clients, self.temp_selected_clients = self.select_clients()
#                 self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients,self.selected_clients)
#                 print("drop classif com")
#                 self.send_models()

#                 if i % self.eval_gap == 0:
#                     print(f"\n-------------Round number: {i}-------------")
#                     print("\nEvaluate global model")
#                     self.evaluate()

#                 print("clients:", len(self.selected_clients))

#                 cnt = 1
#                 for client in self.selected_clients:
#                     st = time.time()
#                     client.train()
#                     train_accuracy = client.get_train_accuracy()
#                     learning_time = time.time() - st
#                     self.track_client_learning_time(client, learning_time)
#                     print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
#                     cnt += 1

#                 for i in range(len(dropped_clients)):
#                     dropped_clients[i].set_weights(self.dropped_clients_nearest_clients[i].get_weights())

#                 st = time.time()
#                 self.receive_models(dropout_ratio=0, required_clients=self.temp_selected_clients)
#                 upload_time = time.time() - st

#                 for client in self.temp_selected_clients:
#                     self.track_client_upload_time(client, upload_time / len(self.temp_selected_clients))
#                 self.aggregate_parameters()

#             else:
#                 # Batch training after the initial round
#                 self.classify_clients()
#                 rounds_batch1, rounds_batch2 = self.adjust_rounds_for_batches()
#                 #rounds_batch1 = min(rounds_batch1,self.global_rounds+1-self.restrict_k)
                 
#                 # if rounds_batch2 > 0:
#                 #     print(f"\n-------------Batch 2 Training: Round {i}-------------")
#                 #     self.train_batch(self.batch2, rounds_batch2)
                                  
#                 # if rounds_batch1 > 0:
#                 #     print(f"\n-------------Batch 1 Training: Round {i}-------------")
#                 #     self.train_batch(self.batch1, rounds_batch1)
#                 #  """Train both batch1 and batch2 concurrently using threads."""
#                 # rounds_batch1, rounds_batch2 = self.adjust_rounds_for_batches()

#                 # Create threads for training batch1 and batch2
#                 print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
#                 print(f"Batch 2 clients: {[client.id for client in self.batch2]}")
#                 thread_batch1 = threading.Thread(target=self.train_batch, args=(self.batch1, rounds_batch1))
#                 thread_batch2 = threading.Thread(target=self.train_batch, args=(self.batch2, rounds_batch2))


#                 # Start both threads
#                 thread_batch1.start()
#                 thread_batch2.start()

#                 # Wait for both threads to complete
#                 thread_batch1.join()
#                 thread_batch2.join()
                 

                

#                 # Aggregate global model after batch training
#                 self.send_models()
                
#                 print("\nEvaluate global model")
#                 self.evaluate()

#                 st = time.time()
#                 self.receive_models(dropout_ratio=self.temp_droupout_ratio, required_clients=self.temp_selected_clients)
#                 upload_time = time.time() - st

#                 for client in self.temp_selected_clients:
#                     self.track_client_upload_time(client, upload_time / len(self.temp_selected_clients))
#                 self.aggregate_parameters()

#                 e_t = time.time()
#                 print(f"Round time cost: {e_t - s_t}")
#                 break   



import time
import random
import numpy as np
from flcore.servers.serverbase import Server
from flcore.clients.clientavg import clientAVG
import threading
import wandb

wandb.init(project='rounds', entity='naveen2112')

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.temp = 1
        self.temp_droupout_ratio = 0

        self.set_slow_clients()  # Assuming this method exists
        self.set_clients(clientAVG)  # Assuming this method sets up clients
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.client_learning_times = {}
        self.client_upload_times = {}
        self.restrict_k=20;
        self.dropout_ratio_dr=0

    def select_clients(self):
        """Select a fraction of clients based on join_ratio and dropout_rate."""
        num_selected_clients = max(1, int(self.join_ratio * self.num_clients))
        available_clients = random.sample(self.clients, num_selected_clients)

        selected_clients = []
        temp_selected_clients = []
        dropped_clients = []
        dropout_rate =  self.dropout_ratio_dr  # Example dropout rate, adjust as needed

        temp_selected_clients = available_clients

        threshold_val = dropout_rate * self.temp * 20
        self.temp_droupout_ratio = dropout_rate * self.temp

        a = 0
        for client in temp_selected_clients:
            a += 1
            if a > threshold_val:
                selected_clients.append(client)
            else:
                dropped_clients.append(client)

        if len(selected_clients) == 0:
            selected_clients = available_clients
            dropped_clients = []
            self.temp_droupout_ratio = 0

        print("Dropped clients:", [client.id for client in dropped_clients])

        selected_clients.sort(key=lambda client: client.id)
        dropped_clients.sort(key=lambda client: client.id)
        temp_selected_clients.sort(key=lambda client: client.id)

        return selected_clients, dropped_clients, temp_selected_clients

    def euclidean_distance(self, weights1, weights2):
        """Calculate the Euclidean distance between two sets of weights."""
        w1 = np.concatenate([v.flatten() for v in weights1.values()])
        w2 = np.concatenate([v.flatten() for v in weights2.values()])
        return np.linalg.norm(w1 - w2)

    def ordered_dict_to_array(self, ordered_dict):
        """Convert OrderedDict of weights to numpy array."""
        return np.concatenate([value.flatten() for value in ordered_dict.values()])

    def cosine_similarity(self, vec1, vec2):
        """Calculate the cosine similarity between two weight vectors."""
        vec1_array = self.ordered_dict_to_array(vec1)
        vec2_array = self.ordered_dict_to_array(vec2)
        dot_product = np.dot(vec1_array, vec2_array)
        norm_vec1 = np.linalg.norm(vec1_array)
        norm_vec2 = np.linalg.norm(vec2_array)
        return dot_product / (norm_vec1 * norm_vec2)

    def find_nearest_trained_client(self, dropped_client, trained_clients):
        """Find the nearest trained client to a dropped client based on cosine similarity."""
        max_similarity = -1000
        nearest_client = None
        temp_c = None
        dropped_client_weights = dropped_client.get_weights()  # Assuming this method exists

        for client in trained_clients:
            trained_client_weights = client.get_weights()  # Assuming this method exists
            similarity = self.cosine_similarity(dropped_client_weights, trained_client_weights)
            temp_c = client
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_client = client
        if nearest_client is None:
            nearest_client = temp_c
        return nearest_client

    def finding_nearest_clients_to_dropped_client(self, dropped_clients,rem_client):
        """Find the nearest trained clients for each dropped client."""
        dropped_clients_nearest_clients = []
        for dropped_client in dropped_clients:
            nearest_client = self.find_nearest_trained_client(dropped_client, rem_client)
            if nearest_client:
                dropped_clients_nearest_clients.append(nearest_client)
        return dropped_clients_nearest_clients

    def track_client_learning_time(self, client, learning_time):
        """Track the learning time of a client."""
        if client.id in self.client_learning_times:
            self.client_learning_times[client.id].append(learning_time)
        else:
            self.client_learning_times[client.id] = [learning_time]

    def track_client_upload_time(self, client, upload_time):
        """Track the upload time of a client."""
        if client.id in self.client_upload_times:
            self.client_upload_times[client.id].append(upload_time)
        else:
            self.client_upload_times[client.id] = [upload_time]

    def classify_clients(self):
        """Classify clients into two batches based on their learning and upload speeds."""
        learning_speed = [(client, np.mean(self.client_learning_times.get(client.id, [0]))) for client in self.clients]
        upload_speed = [(client, np.mean(self.client_upload_times.get(client.id, [0]))) for client in self.clients]

        # Sort clients based on average learning time
        learning_speed.sort(key=lambda x: x[1])

        # Sort clients based on average upload time
        upload_speed.sort(key=lambda x: x[1])

        # Extract slow learners and slow uploaders
        slow_learners = [x[0] for x in learning_speed[:len(learning_speed) // 2]]
        slow_uploaders = [x[0] for x in upload_speed[:len(upload_speed) // 2]]

        # Create batches: Batch 1 contains slow learners and uploaders, Batch 2 contains others
        self.batch1 = list(set(slow_learners + slow_uploaders))
        self.batch2 = [client for client in self.clients if client not in self.batch1]

        print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
        print(f"Batch 2 clients: {[client.id for client in self.batch2]}")

    def calculate_average_time_per_round(self, batch_clients):
        """Calculate the average time per round for a batch of clients."""
        total_learning_time = sum(np.mean(self.client_learning_times.get(client.id, [0])) for client in batch_clients)
        total_upload_time = sum(np.mean(self.client_upload_times.get(client.id, [0])) for client in batch_clients)
        avg_learning_time = total_learning_time / len(batch_clients)
        avg_upload_time = total_upload_time / len(batch_clients)
        return avg_learning_time + avg_upload_time

    def adjust_rounds_for_batches(self):
        """Adjust the number of rounds for each batch to optimize total training time."""
        print("hello")
        avg_time_batch1 = self.calculate_average_time_per_round(self.batch1)
        avg_time_batch2 = self.calculate_average_time_per_round(self.batch2)

        total_rounds = self.global_rounds - self.restrict_k         # Remaining rounds after initial 5 rounds
        total_time = (avg_time_batch1 + avg_time_batch2) / 2 * total_rounds

        rounds_batch1 = int(total_time / avg_time_batch1)
        rounds_batch2 = int(total_time / avg_time_batch2)

        return rounds_batch1, rounds_batch2
    def train_batch_threaded(self, batch, rounds):
        """Thread function to train a batch for specified rounds."""
        for _ in range(rounds):
            # Example logic, replace with your actual training process
            time.sleep(1)  # Simulate training time
            print(f"Training {batch} - round {_+1}")


    
    def train_client(self, client):
        st = time.time()
        client.train()  # Replace with actual training logic
        train_accuracy = client.get_train_accuracy()  # Assuming this method exists
        learning_time = time.time() - st
        self.track_client_learning_time(client, learning_time)
        print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
        

    def train_batch(self, batch_clients, num_rounds):
        """Train a specific batch of clients for a given number of rounds."""
        for round_num in range(num_rounds):
            print(f"Training round {round_num+1}/{num_rounds} for batch {batch_clients[0].id}")
            #selected_clients, dropped_clients, temp_selected_clients = self.select_clients()
            # self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients,selected_clients)
            # batch_clients = random.shuffle(batch_clients)
            total_clients = random.sample(batch_clients,len(batch_clients))
            print(type(batch_clients))
            dr= self.dropout_ratio_dr
            drop_len = dr*len(batch_clients)
            print("."*25,drop_len,"."*25)
            friend_client=[]
            drop_client=[]
            a=1;
            for client in total_clients:
              if(a>drop_len):
                   friend_client.append(client)
              else:
                  drop_client.append(client)
              a=a+1    

            drop_client_friend_client =self.finding_nearest_clients_to_dropped_client(drop_client,friend_client)      

            self.send_models()#set weights

            threads = []
            for client in friend_client:
                thread = threading.Thread(target=self.train_client, args=(client,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            print("."*25, "leng of drop and ffrnd")
            print(len(drop_client))
            print(len(drop_client_friend_client) )
            for i in range(min(len(drop_client),len(drop_client_friend_client))):
                drop_client[i].set_weights(drop_client_friend_client[i].get_weights())

            st = time.time()
            self.receive_models(dropout_ratio=0, required_clients=self.clients)
            upload_time = time.time() - st
            for client in self.clients:
                self.track_client_upload_time(client, upload_time / len(self.clients))
            self.aggregate_parameters()

            self.send_models()
            print("\nEvaluate global model")
            self.evaluate()

    def train(self):
        """Train the federated averaging model."""
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            if i <self.restrict_k :
                # Initial round: Train all clients
                self.selected_clients, dropped_clients, self.temp_selected_clients = self.select_clients()
                self.dropped_clients_nearest_clients = self.finding_nearest_clients_to_dropped_client(dropped_clients,self.selected_clients)
                print("drop classif com")
                self.send_models()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate()

                print("clients:", len(self.selected_clients))

                cnt = 1
                for client in self.selected_clients:
                    st = time.time()
                    client.train()
                    train_accuracy = client.get_train_accuracy()
                    learning_time = time.time() - st
                    self.track_client_learning_time(client, learning_time)
                    print(f"Training client: {client.id}, Time Cost: {learning_time}, Train Accuracy: {train_accuracy:.2f}%")
                    cnt += 1

                # for i in range(len(dropped_clients)):
                #     dropped_clients[i].set_weights(self.dropped_clients_nearest_clients[i].get_weights())

                st = time.time()
                self.receive_models(dropout_ratio=0, required_clients=self.selected_clients)
                upload_time = time.time() - st

                for client in self.temp_selected_clients:
                    self.track_client_upload_time(client, upload_time / len(self.temp_selected_clients))
                self.aggregate_parameters()

            else:
                # Batch training after the initial round
                self.classify_clients()
                rounds_batch1, rounds_batch2 = self.adjust_rounds_for_batches()
                #rounds_batch1 = min(rounds_batch1,self.global_rounds+1-self.restrict_k)
                 
                # if rounds_batch2 > 0:
                #     print(f"\n-------------Batch 2 Training: Round {i}-------------")
                #     self.train_batch(self.batch2, rounds_batch2)
                                  
                # if rounds_batch1 > 0:
                #     print(f"\n-------------Batch 1 Training: Round {i}-------------")
                #     self.train_batch(self.batch1, rounds_batch1)
                #  """Train both batch1 and batch2 concurrently using threads."""
                # rounds_batch1, rounds_batch2 = self.adjust_rounds_for_batches()

                # Create threads for training batch1 and batch2
                print(f"Batch 1 clients: {[client.id for client in self.batch1]}")
                print(f"Batch 2 clients: {[client.id for client in self.batch2]}")
                thread_batch1 = threading.Thread(target=self.train_batch, args=(self.batch1, rounds_batch1))
                thread_batch2 = threading.Thread(target=self.train_batch, args=(self.batch2, rounds_batch2))


                # Start both threads
                thread_batch1.start()
                thread_batch2.start()

                # Wait for both threads to complete
                thread_batch1.join()
                thread_batch2.join()
                 

                

                # Aggregate global model after batch training
                self.send_models()
                
                print("\nEvaluate global model")
                self.evaluate()

                st = time.time()
                self.receive_models(dropout_ratio=self.temp_droupout_ratio, required_clients=self.temp_selected_clients)
                upload_time = time.time() - st

                for client in self.temp_selected_clients:
                    self.track_client_upload_time(client, upload_time / len(self.temp_selected_clients))
                self.aggregate_parameters()

                e_t = time.time()
                print(f"Round time cost: {e_t - s_t}")
                break   