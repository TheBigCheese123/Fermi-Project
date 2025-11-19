# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:41:39 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pyro
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import seaborn as sns
import pandas as pd
import torchvision.transforms as transforms

import idx2numpy
from pyro.contrib.examples.util import MNIST
import pyro.contrib.examples.util

pyro.set_rng_seed(1)

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

# for loading and batching MNIST dataset
def setup_data_loaders(train_set, test_set, batch_size=128, use_cuda=False):
    
    #kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True)# **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False)#, **kwargs)
    
    return train_loader, test_loader

flatten = nn.Flatten()

temp = idx2numpy.convert_from_file('data\\MNIST\\train-images-idx3-ubyte')
train_set = torch.from_numpy(temp)
train_labels = idx2numpy.convert_from_file('data\\MNIST\\train-labels-idx1-ubyte')

temp = idx2numpy.convert_from_file('data\\MNIST\\t10k-images-idx3-ubyte')
test_set = torch.from_numpy(temp)
test_labels = idx2numpy.convert_from_file('data\\MNIST\\t10k-labels-idx1-ubyte')
temp=0

train_loader, test_loader = setup_data_loaders(train_set, test_set)
yes = next(iter(train_loader))

labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

input_dim = 784
output_dim = 10

test = pyro.nn.DenseNN(input_dim, [512], param_dims=[output_dim], nonlinearity=nn.Softmax(dim=10))
test.flatten = nn.Flatten()
print(test)

class TestNetwork(PyroModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            )
    def forward(self, input_data):
        input_data = self.flatten(input_data)
        logits = self.model_stack(input_data)
        return logits

pls_work = TestNetwork(28*28, 10)
print(pls_work)
