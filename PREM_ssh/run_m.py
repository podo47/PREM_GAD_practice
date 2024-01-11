import os
import argparse
import torch
import numpy as np

# import matplotlib.pyplot as plt
from modules.model import Dataloader
from modules.utils import load_dataset
from modules.experiment import run_experiment

parser = argparse.ArgumentParser(description='GGD Anomaly')
parser.add_argument('--dataset', type=str, default='Flickr')

parser.add_argument('--data_type', type=str, choices=['prem','inject'], default='inject')

parser.add_argument('--type', type=str, choices=['all','str','attr'], default='all')

parser.add_argument('--count', type=int, default=10)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--gamma', type=float, default=0.4)

parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--k', type=int, default=2)

parser.add_argument('--resultdir', type=str, default='results')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_epoch', type=int, default=1500)
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--batch_size', type=int, default=-1)

if __name__ == '__main__':
    args = parser.parse_args()
    # Setup torch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    seed = args.seed

    total_auc = []
    total_train_time = []
    total_train_mem = []
    total_test_time = []
    total_test_mem = []

    for i in range(args.count):
        # Load dataset
        g, features, all_ano_label, str_ano_label, attr_ano_label = load_dataset(args, args.dataset)
        features = torch.FloatTensor(features)

        if args.batch_size == -1:
            features = features.to(device)
        #else:   
            #features = features.to(device)

        g = g.to(device)
        dataloader = Dataloader(g, features, args.k, dataset_name=args.dataset)
        if not os.path.isdir("./ckpt"):
            os.makedirs("./ckpt")

        # Run the experiment
        seed += 5

        ano_label = eval(args.type + "_ano_label")

        model, stats = run_experiment(args, seed, device, dataloader, ano_label, all_ano_label)

        # Save the experiment result
        total_auc.append(stats["AUC"])
        total_train_time.append(stats["time_train"])
        total_train_mem.append(stats["mem_train"])
        total_test_time.append(stats["time_test"])
        total_test_mem.append(stats["mem_test"])
    
    np.mean(total_auc)
    print("AUC: %.4f" % np.mean(total_auc))
    print("Time (Train): %.4fs" % np.mean(total_train_time))
    print("Mem (Train): %.4f MB" % (np.mean(total_train_mem) / 1024 / 1024))
    print("Time (Test): %.4fs" % np.mean(total_test_time))
    print("Mem (Test): %.4f MB" % (np.mean(total_test_mem) / 1024 / 1024))
    exit()