import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from skorch.regressor import NeuralNetRegressor
from xenonpy.descriptor import Compositions
from tqdm import tqdm
import shutil
import os


def descriptor_get():
    # obtain descriptor
    descriptor = pd.read_csv('descriptor.csv')
    prop_all = pd.read_csv('id_prop.csv')
    # prop includes band_gap and density
    prop = prop_all['band_gap']

    # minmaxnormal
    scaler = MinMaxScaler()
    descriptor = scaler.fit_transform(descriptor)
    return descriptor, prop


def train_loaders(descriptor, prop, BATCH_SIZE):
    descriptor = torch.tensor(descriptor)
    prop = torch.tensor(prop)
    dataset = torch.utils.data.TensorDataset(descriptor, prop)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_model(estimator, train_loader, epoches, LR):
    estimator.train()
    best_mae = 1
    opt_adam = torch.optim.Adam(estimator.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    mae_list = []
    print_freq = 1
    for i in range(epoches):
        for step, (b_x, b_y) in enumerate(train_loader, 0):
            b_x = b_x.float()
            b_y = b_y.float()
            b_x, b_y = Variable(b_x), Variable(b_y)
            y_pred = estimator(b_x)
            loss = loss_func(y_pred, b_y)
            opt_adam.zero_grad()
            loss.backward()
            opt_adam.step()
            mae = torch.mean(abs(y_pred - b_y))
            mae_list.append(mae)
            is_best = mae < best_mae
            best_mae = min(mae, best_mae)
            if i % print_freq == 0:
                print(f"epoch:{i}\t", "loss: ", loss, "lr:", opt_adam.param_groups[0]['lr'], "mae: ",mae)
            save_checkpoint({
                'epoch': i + 1,
                'state_dict': estimator.state_dict(),
                'best_mae_error': best_mae,
                'optimizer': opt_adam.state_dict(),
            }, is_best)
    return estimator


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h_layer1 = 198
        self.h_layer2 = 90
        self.h_layer3 = 50
        self.sharedlayer = nn.Sequential(
            nn.Linear(290, self.h_layer1),
            nn.BatchNorm1d(self.h_layer1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.h_layer1, self.h_layer2),
            nn.BatchNorm1d(self.h_layer2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.h_layer2, self.h_layer3),
            nn.BatchNorm1d(self.h_layer3),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.finallayer = nn.Sequential(
            nn.Linear(self.h_layer3, 1)
        )

    def forward(self, x):
        x = self.sharedlayer(x)
        h_shared = self.finallayer(x)
        h_shared = h_shared.squeeze(-1)
        return h_shared


descriptor, prop = descriptor_get()
train_loader = train_loaders(descriptor, prop, 256)
model = Net()
model = train_model(model, train_loader, 300, 0.01, descriptor,prop)
