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

class Client():
    def __init__(self, args, model, dataset, idx):
        # self.args = args
        self.device = args.device
        self.model = copy.deepcopy(model).to(self.device)
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
        self.model_path = args.model_path
        self.window_size = args.window_size
        self.input_size = args.input_size
        self.num_classes = args.num_classes 
        self.num_candidates = args.num_candidates

        if args.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = args.lr, momentum = 0.9)
        elif args.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr, betas = (0.9, 0.99))
        else:
            raise NotImplementedError
        
        self.start_epoch = 0
        self.local_epoch = args.local_epoch
        self.best_loss = 0
        self.best_score = -1

    def train(self, epoch):
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
            (epoch+1, start, lr))
        # self.log['train']['lr'].append(lr)
        # self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            output = self.model(features=features, device=self.device)
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            # loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))


    def valid(self, epoch):
        self.model.eval()
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
                output = self.model(features=features, device=self.device)
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
        print('| Model {} |'.format(self.idx+1))
        for epoch in range(self.start_epoch, self.local_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            self.train(epoch)
            if epoch >= self.local_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
            #     self.save_checkpoint(epoch,
            #                          save_optimizer=True,
            #                          suffix="epoch" + str(epoch))
            # self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            # self.save_log()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
    
    def test(self):
        self.load_model()
        # print('test model')
        self.model.to(self.device)
        self.model.eval()
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
                    output = self.model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
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
                    output = self.model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
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

