import argparse
import os
import pickle
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Parameters
RANDOM_SEED = 12
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 32

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--use_all', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = "cuda:0" if args.gpu >= 0 else "cpu"

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    print('Loading previously saved tensors from .pt files...')
    ds = torch.load(args.dataset)
    train_x = ds['train_x'].to(device)
    train_xp = ds['train_xp'].to(device)
    train_y = ds['train_y'].to(device)
    x = ds['valid_x'].to(device)
    xp = ds['valid_xp'].to(device)
    y = ds['valid_y'].to(device)

    if args.use_all:
        x = torch.cat([train_x, x], 0)
        y = torch.cat([train_y, y], 0)
        xp = torch.cat([train_xp, xp], 0)

    dataset = TensorDataset(x, y, xp)

    torch.no_grad()
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torch.jit.load(args.model).to(device)
    model.eval()

    y_pred = np.zeros(len(loader.dataset))
    y_true = np.zeros(len(loader.dataset))

    print('Running network...')

    for i, [x, y, xp] in enumerate(loader):
        bz = x.shape[0]
        x = x.to(device)
        xp = xp.to(device)

        y_ = model(x, xp)
        y_pred[i:i+bz] = y_[:, 0].detach().cpu().numpy()
        y_true[i:i+bz] = y[:, 0].detach().cpu().numpy()

    output = args.model + "_score_test.pkl"
    pickle.dump(
        {
            "y_pred": y_pred,
            "y_true": y_true
        },
        open(output, "wb"),
        protocol=4
    )

    print('Result written to:', output)
