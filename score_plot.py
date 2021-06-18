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
    args = parser.parse_args()

    data = pickle.load(open(args.data, "br"))

    x = data["y_true"]
    y = data["y_pred"]

    h, _, _ = np.histogram2d(x, y, bins=10, range=[[0, 1], [0, 1]], density=True)

    h[0, 0] = 0

    plt.imshow(np.log(h), extent=[0, 1, 1, 0], cmap="gist_heat")
    plt.xlabel("Predicted Score")
    plt.ylabel("Labeled Score")
    plt.show()

