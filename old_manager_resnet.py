
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()

parser.add_argument('data_root', type=str)
parser.add_argument('--output', type=str, default="train_out")
parser.add_argument('--gpu', type=str, default="-1")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Import Pytorch after setting CUDA environs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from PIL import Image

# Regenerate .pt files for the train and test tensors?
# Setting this to False is faster when reading from cephfs is slow and the tensors were previously written out already
DO_REGENERATE_TENSORS = False

# Parameters
RANDOM_SEED = 12
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.002
BATCH_SIZE = 32
N_EPOCHS = 50
P_DROPOUT = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_IMAGES = True
USE_FEATURES = True


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                  stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                  stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, p_dropout, _use_images, _use_features):
        super(Model, self).__init__()

        self.use_images = _use_images
        self.use_features = _use_features

        w = 8

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(1, w * 1, kernel_size=3, stride=1, padding=1),
            ResidualBlock(w * 1, w * 1),
            nn.Dropout(p=p_dropout),

            nn.Conv2d(w * 1, w * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock(w * 2, w * 2),
            nn.Dropout(p=p_dropout),

            nn.Conv2d(w * 2, w * 4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(w * 4, w * 4),
            nn.Dropout(p=p_dropout),

            nn.Conv2d(w * 4, w * 8, kernel_size=3, stride=2, padding=1),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(w * 8 * 8 * 8, 512)
        )

        self.classifier_images_and_features = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(512 + 24, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(128, 1)
        )

        self.classifier_images = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(512, 1)
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


def load_star(filename):
    from collections import OrderedDict
    # This is not a very serious parser; should be token-based.
    datasets = OrderedDict()
    current_data = None
    current_colnames = None
    in_loop = 0  # 0: outside 1: reading colnames 2: reading data

    for line in open(filename):
        line = line.strip()

        # remove comments
        comment_pos = line.find('#')
        if comment_pos > 0:
            line = line[:comment_pos]

        if line == "":
            continue

        if line.startswith("data_"):
            in_loop = 0

            data_name = line[5:]
            current_data = OrderedDict()
            datasets[data_name] = current_data

        elif line.startswith("loop_"):
            current_colnames = []
            in_loop = 1

        elif line.startswith("_"):
            if in_loop == 2:
                in_loop = 0

            elems = line[1:].split()
            if in_loop == 1:
                current_colnames.append(elems[0])
                current_data[elems[0]] = []
            else:
                current_data[elems[0]] = elems[1]

        elif in_loop > 0:
            in_loop = 2
            elems = line.split()
            assert len(elems) == len(current_colnames)
            for idx, e in enumerate(elems):
                current_data[current_colnames[idx]].append(e)

    return datasets


def load_mrc(filename, maxz=-1):
    import numpy as np

    inmrc = open(filename, "rb")
    header_int = np.fromfile(inmrc, dtype=np.uint32, count=256)
    inmrc.seek(0, 0)
    header_float = np.fromfile(inmrc, dtype=np.float32, count=256)

    nx, ny, nz = header_int[0:3]
    eheader = header_int[23]
    mrc_type = None
    if header_int[3] == 2:
        mrc_type = np.float32
    elif header_int[3] == 6:
        mrc_type = np.uint16
    if maxz > 0:
        nz = np.min([maxz, nz])
    # print "NX = %d, NY = %d, NZ = %d" % (nx, ny, nz), mrc_type

    inmrc.seek(1024 + eheader, 0)
    map_slice = np.fromfile(inmrc, mrc_type, nx * ny * nz).reshape(nz, ny, nx).astype(np.float32)

    return nx, ny, nz, map_slice


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for X, y_true, XP in train_loader:
        optimizer.zero_grad()

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
    mean_late_loss = np.mean(np.array(valid_losses[-min(len(valid_losses)-1, 10):]))
    print(f'Final valid loss: {mean_late_loss}')

    return model, optimizer, (train_losses, valid_losses)


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


sys.stdout = Logger(args.output + 'std.out')

torch.manual_seed(RANDOM_SEED)
print('DEVICE=', DEVICE)

if (DO_REGENERATE_TENSORS):
    print('Generating tensors from the RELION STAR file...')
    dataset = load_star(args.data_root + '/combined_features_normalized.star')['normalized_features']
    #nr_train = 12884
    #nr_test = 2309
    nr_train = 9979
    nr_test = 4714
    #nr_train = 24
    #nr_test = 10
    nr_entries = len(dataset['rlnClassScore'])
    fn_subimage = dataset['rlnSubImageStack'][0]
    nx, ny, nz, testsubimage = load_mrc(fn_subimage)
    
    my_x_train = np.zeros(shape=(nr_train*nz, 1, nx, ny), dtype=np.single)
    my_x_test = np.zeros(shape=(nr_test*nz, 1, nx, ny), dtype=np.single)
    for i,x in enumerate(dataset['rlnSubImageStack']):
        if i < nr_train + nr_test:
            if (i%1000 == 0):
                print(i)
            nx, ny, nz, subimage = load_mrc(x)
            for z in range(nz):
                if i < nr_train:
                    my_x_train[i*nz + z, 0] = subimage[z,:,:]
                else:
                    my_x_test[(i-nr_train)*nz + z, 0] = subimage[z,:,:]
        else:
            break
    
    my_xp_train = np.zeros(shape=(nr_train*nz,24))
    my_xp_test = np.zeros(shape=(nr_test*nz,24))
    for i,x in enumerate(dataset['rlnNormalizedFeatureVector']):
        stringarray = x.replace('[','').replace(']','').split(',')
        if i < nr_train + nr_test:
            for z in range(nz):
                for j,y in enumerate(stringarray):
                    if (i < nr_train):
                        my_xp_train[i*nz + z, j] =float(y)
                    else:
                        my_xp_test[(i-nr_train)*nz + z, j] = float(y)
    
    my_y_train = np.zeros(shape=(nr_train*nz, 1), dtype=np.single)
    my_y_test = np.zeros(shape=(nr_test*nz, 1), dtype=np.single)
    
    for i,x in enumerate(dataset['rlnClassScore']):
        if i < nr_train + nr_test:
            for z in range(nz):
                score = float(x)
                if (i < nr_train):
                    my_y_train[i*nz + z, 0] = score
                else:
                    my_y_test[(i-nr_train)*nz + z, 0] = score
        else:
            break
    
    print('y_train_shape= ', my_y_train.shape)
    print('x_train_shape= ',my_x_train.shape)
    print('xp_train_shape= ',my_xp_train.shape)
    print('y_test_shape= ', my_y_test.shape)
    print('x_test_shape= ',my_x_test.shape)
    print('xp_test_shape= ',my_xp_test.shape)
    
    x_train_var = torch.Tensor(my_x_train)
    x_test_var = torch.Tensor(my_x_test)
    xp_train_var = torch.Tensor(my_xp_train)
    xp_test_var = torch.Tensor(my_xp_test)
    y_train_var = torch.Tensor(my_y_train)
    y_test_var = torch.Tensor(my_y_test)

    torch.save(x_train_var, args.data_root + '/saved_x_train_var.pt')
    torch.save(xp_train_var, args.data_root + '/saved_xp_train_var.pt')
    torch.save(y_train_var, args.data_root + '/saved_y_train_var.pt')
    torch.save(x_test_var, args.data_root + '/saved_x_test_var.pt')
    torch.save(xp_test_var, args.data_root + '/saved_xp_test_var.pt')
    torch.save(y_test_var, args.data_root + '/saved_y_test_var.pt')
    
else:
    print('Loading previously saved tensors from their .pt files...')
    x_train_var = torch.load(args.data_root + '/saved_x_train_var.pt')
    xp_train_var = torch.load(args.data_root + '/saved_xp_train_var.pt')
    y_train_var = torch.load(args.data_root + '/saved_y_train_var.pt')
    x_test_var = torch.load(args.data_root + '/saved_x_test_var.pt')
    xp_test_var = torch.load(args.data_root + '/saved_xp_test_var.pt')
    y_test_var = torch.load(args.data_root + '/saved_y_test_var.pt')
    nx = 64
    ny = 64
    nz = 8

train_dataset = TensorDataset(x_train_var.to(DEVICE), y_train_var.to(DEVICE), xp_train_var.to(DEVICE))
valid_dataset = TensorDataset(x_test_var.to(DEVICE), y_test_var.to(DEVICE), xp_test_var.to(DEVICE))
criterion = nn.MSELoss()

print('P_DROPOUT= ',P_DROPOUT,' LEARNING_RATE=',LEARNING_RATE,' BATCH_SIZE= ',BATCH_SIZE, ' USE_IMAGES= ', USE_IMAGES, ' USE_FEATURES= ', USE_FEATURES, ' N_EPOCHS= ', N_EPOCHS)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = Model(P_DROPOUT, USE_IMAGES, USE_FEATURES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
print(model)
model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

modelcpu=model.to('cpu')
modelcpu.eval()
imagetensor=torch.zeros([1,1,64,64], dtype=torch.float)
featuretensor=torch.zeros([1,24], dtype=torch.float)
traced_script_module = torch.jit.trace(modelcpu, (imagetensor,featuretensor) )
outputfilename = args.output
if (USE_IMAGES):
    outputfilename= outputfilename + '_images'

if (USE_FEATURES):
    outputfilename= outputfilename + '_features'

outputfilename= outputfilename + '_batch' + str(BATCH_SIZE)
outputfilename= outputfilename + '_rate' + str(LEARNING_RATE)
outputfilename= outputfilename + '_drop' + str(P_DROPOUT)
outputfilename= outputfilename + '_epoch' + str(N_EPOCHS)
outputfilename= outputfilename + '_model.pt'

traced_script_module.save(outputfilename)
print('Written out model as: ', outputfilename)

