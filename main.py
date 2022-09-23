import argparse

from utils.utils import create_model
from FLAlgorithms.servers.server import Server
from FLAlgorithms.clients.client import Client


def init_server(args):
    model = create_model(args)
    server = Server(args, model)
    return server

def main(args):
    server = init_server(args)
    server.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Server
    parser.add_argument("--mode", type=str, default='fedavg')
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--server_lr", type=float, default=1e-1)
    parser.add_argument("--client_num", type=int, default=5)
    parser.add_argument("--global_epoch", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--device", type=str, default='cpu')
    # Sample
    parser.add_argument("--sample", type=str, default='sliding_window')
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--dataset", type=str, default='hdfs')
    parser.add_argument("--window_size", type=int, default=10)
    # Features
    parser.add_argument("--sequentials", type=bool, default=True)
    parser.add_argument("--quantitatives", type=bool, default=False) # True when LogAnomaly
    parser.add_argument("--semantics", type=bool, default=False)
    # Model
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=28)
    # Train
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--accumulation_step", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--local_epoch", type=int, default=1)

    args = parser.parse_args()
    main(args)


