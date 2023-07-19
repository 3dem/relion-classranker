import argparse
import os
import pickle
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pylab as plt

# Parameters
RANDOM_SEED = 12
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 32

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--cmap', type=str, default='gray')
    args = parser.parse_args()

    data = pickle.load(open(args.data, "br"))

    x = data["y_true"]
    y = data["y_pred"]

    h, _, _ = np.histogram2d(x, y, bins=10, range=[[0, 1], [0, 1]], density=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    if args.title is not None:
        fig.suptitle(args.title)

    h_ = np.empty_like(h)
    h_[:] = np.nan
    h_[h > 0] = np.log(h[h > 0])
    ax1.imshow(h_, extent=[0, 1, 1, 0], cmap=args.cmap)
    ax1.set(xlabel='Predicted Score', ylabel='Labeled Score')
    ax1.set_title("Log")

    h_ = h / np.sum(h, axis=1)[:, None]
    # h_[h == 0] = np.nan
    ax2.imshow(h_, extent=[0, 1, 1, 0], cmap=args.cmap)
    ax2.set(xlabel='Predicted Score')
    ax2.set_title("Row normalized")

    plt.show()

