import sys
from utils.utils import fetch_log_datasets
from FLAlgorithms.clients.client import Client

class Server():
    def __init__(self, args, model):
        self.client_num = args.client_num
        self.clients = []

