import gc
import os
from re import S
import sys
import time
import random
import json
from collections import Counter
sys.path.append('../../')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from FLAlgorithms.models.anomaly_detection_models import deeplog, loganomaly

# from logdeep.dataset.log import log_dataset
# from logdeep.dataset.sample import sliding_window, session_window
# from logdeep.tools.predict import generate

class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True, quan=False, sem=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        return log, self.labels[idx]

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r

def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels

def sliding_window(data_dir, total_client_num, client_num, datatype, window_size, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    if datatype == 'train':
        data_dir += 'hdfs/hdfs_train'
    if datatype == 'val':
        data_dir += 'hdfs/hdfs_test_normal'
    
    count = len(open(data_dir,'r').readlines())
    
    with open(data_dir, 'r') as f:
        for line in f.readlines()[int(count/total_client_num)*client_num : int(count/total_client_num)*(client_num + 1)]: # 5 clients
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * 28
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event - 1)])
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels

def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += 'hdfs/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'hdfs/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'hdfs/robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')
        ]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels

def generate(name, total_client_num, client_num):
    window_size = 10
    hdfs = {}
    length = 0
    
    count = len(open('./data/hdfs/' + name,'r').readlines())
    
    with open('./data/hdfs/' + name, 'r') as f:
        for ln in f.readlines()[int(count/total_client_num)*client_num : int(count/total_client_num)*(client_num + 1)]:
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length

def fetch_log_datasets(args, total_client_num, client_num):
    sample = args.sample
    data_dir = args.data_dir
    window_size = args.window_size
    sequentials = args.sequentials
    quantitatives = args.quantitatives
    semantics = args.semantics
    batch_size = args.batch_size
    
    test_loader = {
      'normal': {
        'normal_loader': None,
        'normal_len': None
      },
      'abnormal': {
        'abnormal_loader': None,
        'abnormal_len': None
      }
    }
    if sample == 'sliding_window':
            #print('train_set')
            train_logs, train_labels = sliding_window(data_dir, total_client_num, client_num,
                                                  datatype = 'train',
                                                  window_size = window_size,)
                                                  
            #print('val_set')
            val_logs, val_labels = sliding_window(data_dir, total_client_num, client_num,
                                                  datatype = 'val',
                                                  window_size = window_size,
                                                  sample_ratio = 0.001)
    elif sample == 'session_window':
        train_logs, train_labels = session_window(data_dir,
                                                  datatype='train')
        val_logs, val_labels = session_window(data_dir,
                                              datatype='val')
    else:
        raise NotImplementedError
    
    train_dataset = log_dataset(logs = train_logs,
                                labels = train_labels,
                                seq = sequentials,
                                quan = quantitatives,
                                sem = semantics)

    valid_dataset = log_dataset(logs=val_logs,
                                labels=val_labels,
                                seq = sequentials,
                                quan = quantitatives,
                                sem = semantics)
    

    del train_logs
    del val_logs
    gc.collect()

    train_loader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True)                  
                                                             
    valid_loader = DataLoader(valid_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True)

    test_normal_loader, test_normal_length = generate('hdfs_test_normal', total_client_num, client_num)
    test_abnormal_loader, test_abnormal_length = generate('hdfs_test_abnormal', total_client_num, client_num)
        
    test_loader['normal']['normal_loader'] = test_normal_loader
    test_loader['normal']['normal_len'] = test_normal_length
    test_loader['abnormal']['abnormal_loader'] = test_abnormal_loader
    test_loader['abnormal']['abnormal_len'] = test_abnormal_length  

    return train_loader, valid_loader, test_loader

def create_model(args):
    if args.model.lower() == 'deeplog':
        model = deeplog(input_size = args.input_size, 
                        hidden_size = args.hidden_size, 
                        num_layers = args.num_layers, 
                        num_keys = args.num_keys)
    elif args.model.lower() == 'loganomaly':
        model = loganomaly(input_size = args.input_size, 
                        hidden_size = args.hidden_size, 
                        num_layers = args.num_layers, 
                        num_keys = args.num_keys)
    else:
        raise NotImplementedError
    return model