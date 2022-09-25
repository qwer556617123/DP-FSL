import argparse
import copy
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

from utils.utils import create_model

from FLAlgorithms.servers.server import Server
from FLAlgorithms.servers.SL_server import SL_Server


def init_server(args):
    model = create_model(args)
    if args.split:
        server = SL_Server(args, model)
    else:
        server = Server(args, model)
    return server

def main(args):
    server = init_server(args)
    if args.train:
        server.train()
    else:
        server.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Server
    parser.add_argument("--train", type=bool, default=True) # train if true, else test
    parser.add_argument("--mode", type=str, default='fedavg') # fedavg / fedadam
    parser.add_argument("--split", type=bool, default=True) # SL if true, else FL
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--server_lr", type=float, default=1e-1)
    parser.add_argument("--client_num", type=int, default=5)
    parser.add_argument("--global_epoch", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--device", type=str, default='cuda')
    # Differential Privacy
    parser.add_argument("--dp", type=bool, default=True)
    parser.add_argument("--noise_scale", type=float, default=0.02)
    parser.add_argument("--norm_bound", type=float, default=1.5)
    # Sample
    parser.add_argument("--sample", type=str, default='sliding_window')
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--dataset", type=str, default='hdfs')
    # Features
    parser.add_argument("--sequentials", type=bool, default=True)
    parser.add_argument("--quantitatives", type=bool, default=True) # loganomaly has to be true, not needed for deeplog
    parser.add_argument("--semantics", type=bool, default=False)
    # Model
    parser.add_argument("--model_name", type=str, default='loganomaly') # deeplog or loganomaly
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=28)
    # Train
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--accumulation_step", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--local_epoch", type=int, default=5)
    # Predict
    parser.add_argument("--num_candidates", type=int, default=9)
    parser.add_argument("--model_dir", type=str, default="./models/loganomaly/split/2022-09-25-23-11-05") # replace

    args = parser.parse_args()

    # model = create_model(args)
    # model_test = copy.deepcopy(model[0]).to('cuda')
    # print(next(model_test.parameters()).is_cuda)
    # print(model)
    # date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # print(date)

    # model_path = os.path.join("./models/", args.model_name + "/split/" + date)
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)

    # root = "./models/deeplog/split/2022-09-25-20-08-19"
    # files_list = os.listdir(root)
    # print(os.path.join(root, "dp_" + files_list[0] if args.dp else files_list[0] ))

    main(args)