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
    train_subImageStack = ds['train_subImageStack']
    train_referenceImage = ds['train_referenceImage']

    x = ds['valid_x'].to(device)
    xp = ds['valid_xp'].to(device)
    y = ds['valid_y'].to(device)
    subImageStack = ds['valid_subImageStack']
    referenceImage = ds['valid_referenceImage']

    if args.use_all:
        x = torch.cat([x, train_x], 0)
        y = torch.cat([y, train_y], 0)
        xp = torch.cat([xp, train_xp], 0)
        subImageStack += train_subImageStack
        referenceImage += train_referenceImage

    torch.no_grad()

    model = torch.jit.load(args.model).to(device)
    model.eval()

    y_pred = np.zeros(len(y))

    print('Running network...')
    
    current_pos = 0
    
    while True:
        if current_pos + BATCH_SIZE < y.shape[0]:
            x_ = x[current_pos:current_pos + BATCH_SIZE].to(device)
            xp_ = xp[current_pos:current_pos + BATCH_SIZE].to(device)
            bz = BATCH_SIZE
        else:
            x_ = x[current_pos:].to(device)
            xp_ = xp[current_pos:].to(device)
            bz = len(x_)

        y_ = model(x_, xp_)
        y_pred[current_pos:current_pos+bz] = y_[:, 0].detach().cpu().numpy()
        
        current_pos += BATCH_SIZE
        if current_pos > y.shape[0] -1:
            break
        
    output = args.model + "_score_test.pkl"
    pickle.dump(
        {
            "y_pred": y_pred,
            "y_true": y[:, 0].detach().cpu().numpy(),
            "subImageStack": subImageStack,
            "referenceImage": referenceImage
        },
        open(output, "wb"),
        protocol=4
    )

    print('Result written to:', output)
