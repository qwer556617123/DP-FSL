import sys
from utils.utils import fetch_log_datasets
from FLAlgorithms.clients.client import Client

class Server():
    def __init__(self, args, model):
        self.client_num = args.client_num
        self.clients = []
        self.datasets = {i:{'train_loader': None, 'valid_loader': None, 'test_loader': None} for i in range(self.client_num)}

        print("Total clients: {}".format(self.client_num))

        for i in range(self.client_num):
            print("Client {}".format(i+1))
            train_loader, valid_loader, test_loader = fetch_log_datasets(args, self.client_num, i)
            self.datasets[i]['train_loader'] = train_loader
            self.datasets[i]['valid_loader'] = valid_loader
            self.datasets[i]['test_loader'] = test_loader

        for i in range(self.client_num):
            client = Client(args, model, self.datasets[i], i)
            self.clients.append(client)
        
        print("Finished creating servers")
