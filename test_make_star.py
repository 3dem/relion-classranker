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
    parser.add_argument('--output', type=str, default="test_results.star")
    args = parser.parse_args()

    data = pickle.load(open(args.data, "br"))

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    subImageStack = data["subImageStack"]
    referenceImage = data["referenceImage"]

    with open(args.output + "_model.star", "w") as f:
        f.write(
            "# version 30001\n\n" +
            "data_normalized_features\n\n" +
            "loop_\n" +
            "_rlnReferenceImage #1\n" +
            "_rlnSubImageStack #2\n" +
            "_rlnClassScore #3\n" +
            "_rlnPredictedScore #4\n"
        )

        for i in range(len(referenceImage)):
            f.write(
                referenceImage[i] + " " +
                subImageStack[i] + " " +
                str(y_true[i]) + " " +
                str(y_pred[i]) + "\n"
            )
