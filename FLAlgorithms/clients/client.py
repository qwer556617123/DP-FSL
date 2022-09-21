import gc
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Client():
    def __init__(self, args, model, dataset, idx):
        # self.args = args
        self.device = args.device
        self.model = model.deepcopy(model).to(self.device)
        self.dataset = dataset
        self.idx = idx
        self.batch_size = args.batch_size
        # self.learning_rate = args.learning_rate
        # self.optimizer = args.optimizer
        self.train_loader = dataset['train_loader']
        self.valid_loader = dataset['valid_loader']

        if args.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = args.learning_rate, momentum = 0.9)
        elif args.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate, betas = (0.9, 0.99))
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
            (epoch, start, lr))
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
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
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
            if epoch >= self.max_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
            #     self.save_checkpoint(epoch,
            #                          save_optimizer=True,
            #                          suffix="epoch" + str(epoch))
            # self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            # self.save_log()

