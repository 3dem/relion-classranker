import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TRAINING_DIR, "model.py")
sys.path.append(os.path.dirname(TRAINING_DIR))

from training.model import *

# Parameters
RANDOM_SEED = 12
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 32
P_DROPOUT = 0.3
ROT_AUGMENT = True
FLIP_AUGMENT = True


MASK_FEATURE_IDX = np.array([15, 16, 17, 18, 19, 20])


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def random_rot(x):
    theta = np.random.uniform() * 2 * np.pi
    rot_mat = get_rot_mat(theta)[None, ...].type(x.dtype).to(x.device).repeat(x.shape[0], 1, 1)
    grid = torch.nn.functional.affine_grid(rot_mat, x.size(), align_corners=False).type(x.dtype).to(x.device)
    x = torch.nn.functional.grid_sample(x, grid, align_corners=False)
    return x


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for x, y_true, xp in train_loader:

        if ROT_AUGMENT:
            x = random_rot(x)

        if FLIP_AUGMENT and np.random.choice([0, 1]) == 1:
            x = torch.flip(x, [3])

        optimizer.zero_grad()

        if MASK_FEATURE_IDX is not None:
            xp[:, MASK_FEATURE_IDX] = 0

        x = x.to(device)
        xp = xp.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(x, xp)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * x.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0

    for x, y_true, xp in valid_loader:
        x = x.to(device)
        xp = xp.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(x, xp)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):
        dt = time.time()

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            dt = time.time() - dt

            print(f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Time: {dt:.2f}\t')

    # plot_losses(train_losses, valid_losses)
    mean_count = min(len(valid_losses) - 1, 10)
    vloss = np.mean(np.array(valid_losses[-mean_count:]))
    tloss = np.mean(np.array(train_losses[-mean_count:]))
    print(f'Final valid loss: {vloss} (mean of last {mean_count} epochs)')
    print(f'Final train loss: {tloss} (mean of last {mean_count} epochs)')

    return model, (train_losses, valid_losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str)
    parser.add_argument('--output', type=str, default="train_out")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--final-train', action="store_true",
                        help="Use both validation and training dataset to train.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = "cuda:0" if args.gpu >= 0 else "cpu"
    
    sys.stdout = Logger(args.output + '_std.out')

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    print('DEVICE=', device)
    
    print('Loading previously saved tensors from their .pt files...')
    ds = torch.load(args.dataset)
    train_x = ds['train_x'].to(device)
    train_xp = ds['train_xp'].to(device)
    train_y = ds['train_y'].to(device)
    valid_x = ds['valid_x'].to(device)
    valid_xp = ds['valid_xp'].to(device)
    valid_y = ds['valid_y'].to(device)
    
    if args.final_train:
        train_x = torch.cat([valid_x, train_x], 0)
        train_y = torch.cat([valid_y, train_y], 0)
        train_xp = torch.cat([valid_xp, train_xp], 0)

    train_dataset = TensorDataset(train_x, train_y, train_xp)
    valid_dataset = TensorDataset(valid_x, valid_y, valid_xp)
    criterion = nn.MSELoss()
    
    print('P_DROPOUT= ', P_DROPOUT, ' LEARNING_RATE=', LEARNING_RATE, ' BATCH_SIZE= ', BATCH_SIZE, ' USE_IMAGES= ',
          USE_IMAGES, ' USE_FEATURES= ', USE_FEATURES, ' N_EPOCHS= ', args.epochs, 'CNN_W= ', CNN_W,  ' FEAT_EXT_W= ',
          FEAT_EXT_W, ' FEAT_EXT_W2= ', FEAT_EXT_W2)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(P_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(model)

    model, (train_losses, valid_losses) = training_loop(
        model, criterion, optimizer, train_loader, valid_loader, args.epochs, device)
    
    model = model.to('cpu')
    model.eval()

    with open(MODEL_PATH, "r") as file:
        model_definition = file.read()

    output_fn = args.output + '_checkpoint.ckpt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_definition': model_definition,
            'P_DROPOUT': P_DROPOUT,
            'LEARNING_RATE': LEARNING_RATE,
            'BATCH_SIZE': BATCH_SIZE,
            'USE_IMAGES': USE_IMAGES,
            'USE_FEATURES': USE_FEATURES,
            'N_EPOCHS': args.epochs,
            'CNN_W': CNN_W,
            'FEAT_EXT_W': FEAT_EXT_W,
            'FEAT_EXT_W2': FEAT_EXT_W2,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
        },
        output_fn
    )
    print('Written model checkpoint file to: ', output_fn)
