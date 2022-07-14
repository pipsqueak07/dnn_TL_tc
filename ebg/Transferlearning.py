import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_predict
from  sklearn.preprocessing import MinMaxScaler
descriptor=pd.read_csv(r'D:\pythonProject\summary\descriptor_get\tc transfer\descriptors.csv')
prop=pd.read_csv(r'D:\pythonProject\summary\descriptor_get\tc transfer\id_prop.csv',header=None)
prop=prop[1]


scaler=MinMaxScaler()
X_regr=np.array(descriptor).astype(np.float32)
X_regr=scaler.fit_transform(X_regr)
y_regr=np.array(prop).astype(np.float32)
y_regr=y_regr.reshape(-1,1)


#density TL
model=torch.load(r'D:\pythonProject\summary\descriptor_get\model_density.pth.tar')
Model=model.copy()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h_layer1 = 159
        self.h_layer2 = 74
        self.h_layer3 = 43
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
        self.sharedlayer[0].weight = torch.nn.Parameter(Model['state_dict']['sharedlayer.0.weight'])
        # self.sharedlayer[0].weight.requires_grad = False
        self.sharedlayer[0].bias = torch.nn.Parameter(Model['state_dict']['sharedlayer.0.bias'])
        # self.sharedlayer[0].bias.requires_grad = False
        self.sharedlayer[4].weight = torch.nn.Parameter(Model['state_dict']['sharedlayer.4.weight'])
        # self.sharedlayer[0].weight.requires_grad = False
        self.sharedlayer[4].bias = torch.nn.Parameter(Model['state_dict']['sharedlayer.4.bias'])
        # self.sharedlayer[0].bias.requires_grad = False
        self.sharedlayer[8].weight = torch.nn.Parameter(Model['state_dict']['sharedlayer.8.weight'])
        # self.sharedlayer[0].weight.requires_grad = False
        self.sharedlayer[8].bias = torch.nn.Parameter(Model['state_dict']['sharedlayer.8.bias'])
        # self.sharedlayer[0].bias.requires_grad = False
        self.finallayer = nn.Sequential(
            nn.Linear(self.h_layer3, 1)
        )

    def forward(self, x):
        x = self.sharedlayer(x)
        h_shared = self.finallayer(x)
        return h_shared


#ebg TL
model=torch.load(r'D:\pythonProject\summary\descriptor_get\ebg\model_ebg.pth.tar')
Model=model.state_dict().copy()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h_layer1 = 190
        self.h_layer2 = 75
        self.h_layer3 = 20
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
        self.sharedlayer[0].weight = torch.nn.Parameter(Model['sharedlayer.0.weight'])
        # self.sharedlayer[0].weight.requires_grad = False
        self.sharedlayer[0].bias = torch.nn.Parameter(Model['sharedlayer.0.bias'])
        # self.sharedlayer[0].bias.requires_grad = False
        self.sharedlayer[4].weight = torch.nn.Parameter(Model['sharedlayer.4.weight'])
        # self.sharedlayer[0].weight.requires_grad = False
        self.sharedlayer[4].bias = torch.nn.Parameter(Model['sharedlayer.4.bias'])
        # self.sharedlayer[0].bias.requires_grad = False
        self.sharedlayer[8].weight = torch.nn.Parameter(Model['sharedlayer.8.weight'])
        # self.sharedlayer[0].weight.requires_grad = False
        self.sharedlayer[8].bias = torch.nn.Parameter(Model['sharedlayer.8.bias'])
        # self.sharedlayer[0].bias.requires_grad = False
        self.finallayer = nn.Sequential(
            nn.Linear(self.h_layer3, 1)
        )

    def forward(self, x):
        x = self.sharedlayer(x)
        h_shared = self.finallayer(x)
        return h_shared

from skorch.regressor import NeuralNetRegressor
net_regr = NeuralNetRegressor(
    Net,
    max_epochs=500,
    optimizer=torch.optim.Adam,
    optimizer__weight_decay=0,
    batch_size=256,
    lr=0.01,
    train_split=None,
)

predicted = cross_val_predict(net_regr, X_regr, y_regr, cv=5)
mae = mean_absolute_error(y_regr, predicted)
r2 = r2_score(y_regr, predicted)
