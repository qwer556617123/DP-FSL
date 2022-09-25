import sys
import torch
import copy
import os
import datetime

from utils.utils import fetch_log_datasets, clip_generator, add_noise_generator
from FLAlgorithms.clients.SL_client import SL_Client

class SL_Server():
    def __init__(self, args, models):
        self.device = args.device
        self.client_num = args.client_num
        self.clients = []
        self.mode = args.mode.lower()
        self.server_model_1, self.server_model_2 = copy.deepcopy(models[0]).to(self.device), copy.deepcopy(models[1]).to(self.device)
        self.global_epoch = args.global_epoch
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.tau = args.tau
        self.server_lr = args.server_lr
        self.model_name = args.model_name # deeplog or loganomaly

        self.datasets = {i:{'train_loader': None, 'valid_loader': None, 'test_loader': None} for i in range(self.client_num)}

        self.client_weights = [1/self.client_num for i in range(self.client_num)]
        
        # Differential Privacy
        self.dp = args.dp
        self.group = {'noise_scale':args.noise_scale, 'norm_bound':args.norm_bound}

        self.v_1 = {}
        self.grad_1 = {}
        self.v_2 = {}
        self.grad_2 = {}

        for key in self.server_model_1.state_dict().keys():
            self.v_1[key] = torch.add(torch.zeros_like(self.server_model_1.state_dict()[key],dtype=torch.float32),self.tau**2)
            self.grad_1[key] = torch.zeros_like(self.server_model_1.state_dict()[key],dtype=torch.float32)
        for key in self.server_model_2.state_dict().keys():
            self.v_2[key] = torch.add(torch.zeros_like(self.server_model_2.state_dict()[key],dtype=torch.float32),self.tau**2)
            self.grad_2[key] = torch.zeros_like(self.server_model_2.state_dict()[key],dtype=torch.float32)

        print("SL Server")
        print("Model: {}".format(self.model_name))
        print("Mode: {}".format(self.mode))
        print("Total clients: {}".format(self.client_num))

        for i in range(self.client_num):
            print("Client {}".format(i+1))
            train_loader, valid_loader, test_loader = fetch_log_datasets(args, self.client_num, i)
            self.datasets[i]['train_loader'] = train_loader
            self.datasets[i]['valid_loader'] = valid_loader
            self.datasets[i]['test_loader'] = test_loader

        for i in range(self.client_num):
            client = SL_Client(args, models, self.datasets[i], i)
            self.clients.append(client)
        
        print("Finished creating server")

    def train(self):
        for glob_iter in range(self.global_epoch):
            print("\n\n-------------Global Round: ",glob_iter+1, " -------------\n\n")
            for client in self.clients: 
                client.start_train()
            self.aggregate()
        self.save_model()
    
    def aggregate(self):
        # if self.mode.lower() == 'fedbn':
        #     with torch.no_grad():
        #         for key in server_model_1.state_dict().keys():
        #             if 'bn' not in key: #非BN層都去交換
        #                 temp = torch.zeros_like(server_model_1.state_dict()[key], dtype=torch.float32)
        #                 for client_idx in range(client_num):
        #                     temp += client_weights[client_idx] * models_1[client_idx].state_dict()[key]
        #                 server_model_1.state_dict()[key].data.copy_(temp)
        #                 for client_idx in range(client_num):
        #                     models_1[client_idx].state_dict()[key].data.copy_(server_model_1.state_dict()[key])
        #         for key in server_model_2.state_dict().keys():
        #             if 'bn' not in key: #非BN層都去交換
        #                 temp = torch.zeros_like(server_model_2.state_dict()[key], dtype=torch.float32)
        #                 for client_idx in range(client_num):
        #                     temp += client_weights[client_idx] * models_2[client_idx].state_dict()[key]
        #                 server_model_2.state_dict()[key].data.copy_(temp)
        #                 for client_idx in range(client_num):
        #                     models_2[client_idx].state_dict()[key].data.copy_(server_model_2.state_dict()[key])
        
        # elif self.mode.lower() == 'fedadagrad':
        #     with torch.no_grad():
        #         for key, param in server_model_1.named_parameters():
        #             temp = torch.zeros_like(server_model_1.state_dict()[key])
        #             for client_idx in range(len(client_weights)):
        #                 temp = temp + client_weights[client_idx] * models_1[client_idx].state_dict()[key]                          
        #             param.grad = temp - param.data              
        #             param.grad = torch.mul(grad_1[key], args.beta_1) + torch.mul(param.grad, 1-args.beta_1)
        #             grad_1[key] = param.grad
        #             v_1[key] = v_1[key] + param.grad**2
        #             param.data = param.data + torch.mul(torch.div(param.grad, torch.add(torch.sqrt(v_1[key]),args.tau)), args.server_lr) 

        #             for client_idx in range(len(client_weights)):
        #                 models_1[client_idx].state_dict()[key].data.copy_(server_model_1.state_dict()[key])                         
        #         for key, param in server_model_2.named_parameters():
        #             temp = torch.zeros_like(server_model_2.state_dict()[key])
        #             for client_idx in range(len(client_weights)):
        #                 temp = temp + client_weights[client_idx] * models_2[client_idx].state_dict()[key]                          
        #             param.grad = temp - param.data              
        #             param.grad = torch.mul(grad_2[key], args.beta_1) + torch.mul(param.grad, 1-args.beta_1)
        #             grad_2[key] = param.grad
        #             v_2[key] = v_2[key] + param.grad**2
        #             param.data = param.data + torch.mul(torch.div(param.grad, torch.add(torch.sqrt(v_2[key]),args.tau)), args.server_lr) 

        #             for client_idx in range(len(client_weights)):
        #                 models_2[client_idx].state_dict()[key].data.copy_(server_model_2.state_dict()[key]) 
        
        if self.mode.lower() == 'fedadam':
            with torch.no_grad():
                for key, param in self.server_model_1.named_parameters():
                    temp = torch.zeros_like(self.server_model_1.state_dict()[key])
                    for client_idx in range(self.client_num):
                        temp += self.client_weights[client_idx] * self.clients[client_idx].model_1.state_dict()[key]                         
                    param.grad = temp - param.data   
                    if self.dp:
                        clip = clip_generator(norm_bound=self.group['norm_bound'])
                        add_noise = add_noise_generator(noise_scale=self.group['noise_scale'] * self.group['norm_bound']) 
                        param.grad = add_noise(clip(param.grad))
                    param.grad = torch.mul(self.grad_1[key], self.beta_1) + torch.mul(param.grad, 1-self.beta_1) 
                    self.grad_1[key] = param.grad    
                    
                    self.v_1[key] = torch.mul(self.v_1[key], self.beta_2) + torch.mul(param.grad**2, 1-self.beta_2)
                    param.data = param.data + torch.mul(torch.div(param.grad, torch.add(torch.sqrt(self.v_1[key]),self.tau)), self.server_lr)
                    
                    for client_idx in range(self.client_num):
                        self.clients[client_idx].model_1.state_dict()[key].data.copy_(self.server_model_1.state_dict()[key])
                for key, param in self.server_model_2.named_parameters():       
                    temp = torch.zeros_like(self.server_model_2.state_dict()[key])
                    for client_idx in range(self.client_num):
                        temp += self.client_weights[client_idx] * self.clients[client_idx].model_2.state_dict()[key]                           
                    param.grad = temp - param.data   
                    if self.dp:
                        clip = clip_generator(norm_bound=self.group['norm_bound'])
                        add_noise = add_noise_generator(noise_scale=self.group['noise_scale'] * self.group['norm_bound']) 
                        param.grad = add_noise(clip(param.grad))
                    param.grad = torch.mul(self.grad_2[key], self.beta_1) + torch.mul(param.grad, 1-self.beta_1) 
                    self.grad_2[key] = param.grad                
                    self.v_2[key] = torch.mul(self.v_2[key], self.beta_2) + torch.mul(param.grad**2, 1-self.beta_2)
                    param.data = param.data + torch.mul(torch.div(param.grad, torch.add(torch.sqrt(self.v_2[key]),self.tau)), self.server_lr)

                    for client_idx in range(self.client_num):
                        self.clients[client_idx].model_2.state_dict()[key].data.copy_(self.server_model_2.state_dict()[key])

        else:
            with torch.no_grad():
                for key in self.server_model_1.state_dict().keys():#遇到BN層就直接拿第一個client參數使用
                    # num_batches_tracked is a non trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    if 'num_batches_tracked' in key:
                        self.server_model_1.state_dict()[key].data.copy_(self.clients[0].model_1.state_dict()[key])
                    else:
                        temp = torch.zeros_like(self.server_model_1.state_dict()[key]).to(self.device)
                        for client_idx in range(self.client_num):
                            temp += self.client_weights[client_idx] * self.clients[client_idx].model_1.state_dict()[key]
                        if self.dp:
                            clip = clip_generator(norm_bound=self.group['norm_bound'])
                            add_noise = add_noise_generator(noise_scale=self.group['noise_scale'] * self.group['norm_bound'])
                            temp = add_noise(clip(temp))  
                        self.server_model_1.state_dict()[key].data.copy_(temp)#weight傳給server
                        for client_idx in range(self.client_num):
                            self.clients[client_idx].model_1.state_dict()[key].data.copy_(self.server_model_1.state_dict()[key])
                for key in self.server_model_2.state_dict().keys():#遇到BN層就直接拿第一個client參數使用
                    # num_batches_tracked is a non trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    if 'num_batches_tracked' in key:
                        self.server_model_2.state_dict()[key].data.copy_(self.clients[0].model_2.state_dict()[key])
                    else:
                        temp = torch.zeros_like(self.server_model_2.state_dict()[key]).to(self.device)
                        for client_idx in range(self.client_num):
                            temp += self.client_weights[client_idx] * self.clients[client_idx].model_2.state_dict()[key]                        
                        if self.dp:
                            clip = clip_generator(norm_bound=self.group['norm_bound'])
                            add_noise = add_noise_generator(noise_scale=self.group['noise_scale'] * self.group['norm_bound'])
                            temp = add_noise(clip(temp))  
                        self.server_model_2.state_dict()[key].data.copy_(temp)#weight傳給server
                        for client_idx in range(self.client_num):
                            self.clients[client_idx].model_2.state_dict()[key].data.copy_(self.server_model_2.state_dict()[key])
    
    def save_model(self):
        print("saving model")
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_path = os.path.join("./models/", self.model_name + "/split/" + date)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.server_model_1.state_dict(), os.path.join(model_path, 
                                                                str(self.client_num) + '_Client_' + self.model_name + '_SL' + '_1_' + self.mode + '_dp' + ".pt" if self.dp 
                                                                else str(self.client_num) + '_Client_' + self.model_name + '_SL' + '_1_' + self.mode + ".pt"))
        torch.save(self.server_model_2.state_dict(), os.path.join(model_path, 
                                                                str(self.client_num) + '_Client_' + self.model_name + '_SL' + '_2_' + self.mode + '_dp' + ".pt" if self.dp 
                                                                else str(self.client_num) + '_Client_' + self.model_name + '_SL' + '_2_' + self.mode + ".pt"))
        print("Model saved to path: {}".format(model_path))

    def test(self):
        print("\n\n------------- Test -------------\n\n")
        for idx, client in enumerate(self.clients): 
            print('Client ', idx) 
            client.test()