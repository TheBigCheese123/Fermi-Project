# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:13:19 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import torch
from torch import nn
import pyro
from pyro.nn import PyroModule

pyro.set_rng_seed(1)

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

#####  y = 3x + 2  #####
x_data = torch.linspace(0, 100, 101).unsqueeze(1)
y_data = 3 * x_data + 2

##### Gaussian noise profile #####
sigma = 5
y_data = y_data + sigma*torch.randn_like(y_data)

#x_data = torch.tensor(x_data, dtype=torch.float)
#x_data = x_data.T
#y_data = torch.tensor(y_data, dtype=torch.float)

linear_regression_model = PyroModule[nn.Linear](1,1)

loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(linear_regression_model.parameters(), lr=0.05)
iterations = 1000

def train():
    y_pred = linear_regression_model(x_data)#.squeeze(-1)
    
    loss = loss_fn(y_pred, y_data)
    
    optim.zero_grad()
    
    loss.backward()
    
    optim.step()
    return loss


for j in range(iterations):
    loss = train()
    if (j + 1) % 500 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))


# Inspect learned parameters
print("Learned parameters:")
for name, param in linear_regression_model.named_parameters():
    print(name, param.data.numpy())
    
fig = plt.figure()
ax = fig.gca()
ax.scatter(x_data, y_data, marker='x')
ax.plot(x_data, linear_regression_model(x_data).detach().cpu().numpy(), color='black')