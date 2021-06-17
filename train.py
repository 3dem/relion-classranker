import argparse
import os
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
N_EPOCHS = 50
P_DROPOUT = 0.3
CNN_W = 16
FEAT_EXT_W = 512
USE_IMAGES = True
USE_FEATURES = True
ROT_AUGMENT = True

MASK_FEATURE_IDX = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])


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


class Model(nn.Module):
    def __init__(self, p_dropout, _use_images, _use_features):
        super(Model, self).__init__()

        self.use_images = _use_images
        self.use_features = _use_features

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(1, CNN_W, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=p_dropout),

            nn.Conv2d(CNN_W, CNN_W * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=p_dropout),

            nn.Conv2d(CNN_W * 2, CNN_W * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=p_dropout),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(CNN_W * 4 * 7 * 7, FEAT_EXT_W)
        )

        self.classifier_images_and_features = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(FEAT_EXT_W + 24, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(128, 1)
        )

        self.classifier_images = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(FEAT_EXT_W, 1)
        )

        self.classifier_features = nn.Sequential(
            nn.Linear(24, 12),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(12, 1)
        )

    # Defining the forward pass
    def forward(self, x, y):
        if (self.use_images):
            x = self.cnn_layers(x)
            x = torch.flatten(x, 1)
            x = self.feature_extractor(x)
            if (self.use_features):
                z = torch.cat([x, y], 1)
                return self.classifier_images_and_features(z)
            else:
                return self.classifier_images(x)
        elif (self.use_features):
            return self.classifier_features(y)


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

    for X, y_true, XP in train_loader:

        if ROT_AUGMENT:
            X = random_rot(X)

        optimizer.zero_grad()

        if MASK_FEATURE_IDX is not None:
            XP[:, MASK_FEATURE_IDX] = 0

        X = X.to(device)
        XP = XP.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X, XP)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

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

    for X, y_true, XP in valid_loader:
        X = X.to(device)
        XP = XP.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X, XP)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        dt = time.time()

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
    mean_late_loss = np.mean(np.array(valid_losses[-min(len(valid_losses) - 1, 10):]))
    print(f'Final valid loss: {mean_late_loss}')

    return model, optimizer, (train_losses, valid_losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str)
    parser.add_argument('--output', type=str, default="train_out")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = "cuda:0" if args.gpu >= 0 else "cpu"
    
    sys.stdout = Logger(args.output + 'std.out')

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
    
    train_dataset = TensorDataset(train_x, train_y, train_xp)
    valid_dataset = TensorDataset(valid_x, valid_y, valid_xp)
    criterion = nn.MSELoss()
    
    print('P_DROPOUT= ', P_DROPOUT, ' LEARNING_RATE=', LEARNING_RATE, ' BATCH_SIZE= ', BATCH_SIZE, ' USE_IMAGES= ',
          USE_IMAGES, ' USE_FEATURES= ', USE_FEATURES, ' N_EPOCHS= ', N_EPOCHS)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(P_DROPOUT, USE_IMAGES, USE_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(model)

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, device)
    
    model_cpu = model.to('cpu')
    model_cpu.eval()
    image_tensor = torch.zeros([1, 1, 64, 64], dtype=torch.float)
    feature_tensor = torch.zeros([1, 24], dtype=torch.float)
    traced_script_module = torch.jit.trace(model_cpu, (image_tensor, feature_tensor))
    
    output_fn = args.output
    if (USE_IMAGES):
        output_fn = output_fn + '_images'
    if (USE_FEATURES):
        output_fn = output_fn + '_features'
    
    output_fn += '_batch' + str(BATCH_SIZE) + \
                      '_rate' + str(LEARNING_RATE) + \
                      '_drop' + str(P_DROPOUT) + \
                      '_epoch' + str(N_EPOCHS) + \
                      '_model.pt'
    traced_script_module.save(output_fn)
    print('Written out model as: ', output_fn)
