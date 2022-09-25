import gc
import os
import sys
import time
import copy
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class SL_Client():
    def __init__(self, args, models, dataset, idx):
        # self.args = args
        self.device = args.device
        self.model_1, self.model_2 = copy.deepcopy(models[0]).to(self.device), copy.deepcopy(models[1]).to(self.device)
        self.dataset = dataset
        self.idx = idx
        self.batch_size = args.batch_size
        self.accumulation_step = args.accumulation_step
        self.model_name = args.model_name
        self.mode = args.mode

        self.train_loader = dataset['train_loader']
        self.valid_loader = dataset['valid_loader']
        self.test_normal_loader = dataset['test_loader']['normal']['normal_loader']
        self.test_normal_length = dataset['test_loader']['normal']['normal_len']
        self.test_abnormal_loader = dataset['test_loader']['abnormal']['abnormal_loader']
        self.test_abnormal_length = dataset['test_loader']['abnormal']['abnormal_len']

        # predict
        self.model_dir = args.model_dir
        self.window_size = args.window_size
        self.input_size = args.input_size
        self.num_classes = args.num_classes 
        self.num_candidates = args.num_candidates

        if args.optimizer.lower() == 'sgd':
            self.optimizer_1 = torch.optim.SGD(self.model_1.parameters(), 
                                            lr = args.lr, 
                                            momentum = 0.9)
            self.optimizer_2 = torch.optim.SGD(self.model_2.parameters(), 
                                            lr = args.lr, 
                                            momentum = 0.9)
        elif args.optimizer.lower() == 'adam':
            self.optimizer_1 = torch.optim.Adam(self.model_1.parameters(), 
                                            lr = args.lr, 
                                            betas = (0.9, 0.99))
            self.optimizer_2 = torch.optim.Adam(self.model_2.parameters(), 
                                            lr = args.lr, 
                                            betas = (0.9, 0.99))
        else:
            raise NotImplementedError
        
        self.start_epoch = 0
        self.local_epoch = args.local_epoch
        self.best_loss = 0
        self.best_score = -1

    def train(self, epoch):
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer_1.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
            (epoch+1, start, lr))
        # self.log['train']['lr'].append(lr)
        # self.log['train']['time'].append(start)
        self.model_1.train()
        self.model_2.train()

        self.optimizer_1.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        # num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            
            features_1 = []
            features_2 = []
            for item in features:
                features_1.append(item[:, :int(self.window_size/2), :])  # 從特徵切 不是從batch順序切
                features_2.append(item[:, int(self.window_size/2):, :])

            self.optimizer_2.zero_grad()  

            clientout, clientpreh, clientprec = self.model_1(features = features_1, device=self.device)
            output = self.model_2(features = features_2, hPrevious = clientpreh, cPrevious = clientprec, device=self.device)

            
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward(retain_graph = True)

            if (i + 1) % self.accumulation_step == 0:
                self.optimizer_2.step()
                self.optimizer_1.step()
                self.optimizer_2.zero_grad()
                self.optimizer_1.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))


    def valid(self, epoch):
        self.model_1.eval()
        self.model_2.eval()

        # self.log['valid']['epoch'].append(epoch)
        # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch+1, start))
        # self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)

        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                
                features_1 = []
                features_2 = []
                for item in features:
                    features_1.append(item[:, :int(self.window_size/2), :])  # 從特徵切 不是從batch順序切
                    features_2.append(item[:, int(self.window_size/2):, :])

                clientout, clientpreh, clientprec = self.model_1(features = features_1, device = self.device)
                output = self.model_2(features = features_2, hPrevious = clientpreh, cPrevious = clientprec, device=self.device)
                
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        # self.log['valid']['loss'].append(total_losses / num_batch)

        # if total_losses / num_batch < self.best_loss:
        #     self.best_loss = total_losses / num_batch
        #     self.save_checkpoint(epoch,
        #                          save_optimizer=False,
        #                          suffix="bestloss")

    def start_train(self):
        print('| Client {} |'.format(self.idx+1))
        for epoch in range(self.start_epoch, self.local_epoch):
            if epoch == 0:
                self.optimizer_1.param_groups[0]['lr'] /= 32  
                self.optimizer_2.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer_1.param_groups[0]['lr'] *= 2
                self.optimizer_2.param_groups[0]['lr'] *= 2
            self.train(epoch)
            if epoch >= self.local_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
            #     self.save_checkpoint(epoch,
            #                          save_optimizer=True,
            #                          suffix="epoch" + str(epoch))
            # self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            # self.save_log()

    def load_model(self):
        root = self.model_dir
        model_path = os.listdir(root)

        self.model_1.load_state_dict(torch.load(os.path.join(root, model_path[0])))
        self.model_2.load_state_dict(torch.load(os.path.join(root, model_path[1])))
    
    def test(self):
        self.load_model()
        # print('test model')
        self.model_1.to(self.device)
        self.model_2.to(self.device)
        self.model_1.eval()
        self.model_2.eval()
        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            for line in tqdm(self.test_normal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)

                    clientout, clientpreh, clientprec = self.model_1(features = [seq0[:,:int(self.window_size/2),:], seq1[:,:int(self.window_size/2),:]], device = self.device)
                    output = self.model_2(features=[seq0[:,int(self.window_size/2):,:], seq1[:,int(self.window_size/2):,:]]
                                     , hPrevious = clientpreh, cPrevious = clientprec, device=self.device)

                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        FP += self.test_normal_loader[line]
                        break
        with torch.no_grad():
            for line in tqdm(self.test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)

                    clientout, clientpreh, clientprec = self.model_1(features = [seq0[:,:int(self.window_size/2),:], seq1[:,:int(self.window_size/2),:]], device = self.device)
                    output = self.model_2(features=[seq0[:,int(self.window_size/2):,:], seq1[:,int(self.window_size/2):,:]]
                                     , hPrevious = clientpreh, cPrevious = clientprec, device=self.device)
                    
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        TP += self.test_abnormal_loader[line]
                        break

        # Compute precision, recall and F1-measure
        FN = self.test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        Acc = 100 * (TP + self.test_normal_length - FP) / (self.test_abnormal_length + self.test_normal_length) 
        # self.f1 = F1
        # self.acc = Acc
        # self.precision = P
        # self.recall = R
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('actual positive (TP+FN): {}, actual negative (FP+TN): {}'.format(self.test_abnormal_length, self.test_normal_length))
        print('accuracy: {:.3f}'.format(Acc) )
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

        # return self.precision, self.recall, self.f1, self.acc

