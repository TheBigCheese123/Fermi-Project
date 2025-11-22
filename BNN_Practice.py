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
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
import pyro.distributions as dist
import seaborn as sns
import pandas as pd
import torchvision.transforms as transforms

import idx2numpy
from pyro.contrib.examples.util import MNIST
import pyro.contrib.examples.util

#pyro.set_rng_seed(1)

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return the number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and label at the given index
        image = self.images[idx]
        label = self.labels[idx]
        image = image.unsqueeze(0)  # Add channel dimension (C=1)

        return image, label
    
temp = idx2numpy.convert_from_file('data\\MNIST\\train-images-idx3-ubyte')
temp = np.array(temp, dtype=np.float32)/255.0
train_set = torch.from_numpy(temp)
temp = idx2numpy.convert_from_file('data\\MNIST\\train-labels-idx1-ubyte')
train_labels = torch.from_numpy(temp)

temp = idx2numpy.convert_from_file('data\\MNIST\\t10k-images-idx3-ubyte')
temp = np.array(temp, dtype=np.float32)/255.0
test_set = torch.from_numpy(temp)
temp = idx2numpy.convert_from_file('data\\MNIST\\t10k-labels-idx1-ubyte')
test_labels = torch.from_numpy(temp)
temp=0

#train_set = train_set[0:100]
#train_labels = train_labels[0:100]

train_loader = MNISTDataset(train_set, train_labels)
test_loader = MNISTDataset(test_set, test_labels)

input_dim = 784
hidden_dim = 256
output_dim = 10

#test = pyro.nn.DenseNN(input_dim, [512], param_dims=[output_dim], nonlinearity=nn.Softmax(dim=10))
#test.flatten = nn.Flatten()
#print(test)

class NeuralNetwork(PyroModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model_stack = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](input_dim, hidden_dim),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Linear](hidden_dim, output_dim),
            )
        
        self.Layer1WeightsMean = PyroParam(torch.zeros_like(self.model_stack[0].weight))
        self.Layer1WeightsStd = PyroParam(torch.ones_like(self.model_stack[0].weight), constraint=torch.distributions.constraints.positive)

        self.Layer1BiasMean = PyroParam(torch.zeros_like(self.model_stack[0].bias))
        self.Layer1BiasStd = PyroParam(torch.ones_like(self.model_stack[0].bias), constraint=torch.distributions.constraints.positive)
        
        self.Layer2WeightsMean = PyroParam(torch.zeros_like(self.model_stack[2].weight))
        self.Layer2WeightsStd = PyroParam(torch.ones_like(self.model_stack[2].weight), constraint=torch.distributions.constraints.positive)
        
        self.Layer2BiasMean = PyroParam(torch.zeros_like(self.model_stack[2].bias))
        self.Layer2BiasStd = PyroParam(torch.ones_like(self.model_stack[2].bias), constraint=torch.distributions.constraints.positive)
        
        self.model_stack[0].weight = PyroSample(dist.Normal(self.Layer1WeightsMean, self.Layer1WeightsStd).independent())
        self.model_stack[0].bias = PyroSample(dist.Normal(self.Layer1BiasMean, self.Layer1BiasStd).independent())

        self.model_stack[2].weight = PyroSample(dist.Normal(self.Layer2WeightsMean, self.Layer2WeightsStd).independent())
        self.model_stack[2].bias = PyroSample(dist.Normal(self.Layer2BiasMean, self.Layer2BiasStd).independent())

        
    def forward(self, input_data):
        #print(self.Layer1WeightsMean)
        input_data = self.flatten(input_data)
        return self.model_stack(input_data)

TestNetwork = NeuralNetwork(input_dim, hidden_dim, output_dim)

def LeModel(image_set, image_labels):
    #Layer1Weights = pyro.param('Layer1Weights', dist.Normal(0, 1).expand([input_dim, 512]).to_event(2))
    #Layer1Bias = pyro.param('Layer1Bias', dist.Normal(0, 1).expand([512]).to_event(1))

    #Layer2Weights = pyro.param('Layer2Weights', dist.Normal(0, 1).expand([512, output_dim]).to_event(2))
    #Layer2Bias = pyro.param('Layer2Bias', dist.Normal(0, 1).expand([output_dim]).to_event(1))

    #priors = {'L1W': Layer1Weights, 'L1B': Layer1Bias, 'L2W': Layer2Weights, 'L2B': Layer2Bias}
    
    lhat = nn.functional.log_softmax(TestNetwork(image_set))
    #print(lhat.shape, dist.Categorical(logits=lhat))
    with pyro.plate("results", lhat.shape[0]):
        pyro.sample("obs", dist.Categorical(logits=lhat), obs=image_labels)
        #print("Preds:", sampled_value, "Actual:", image_labels)

def CustomGuide(image_set, image_labels):
    Layer1WeightsMean = pyro.param('Layer1WeightsMean', torch.zeros_like(TestNetwork.model_stack[0].weight))
    Layer1WeightsStd = pyro.param('Layer1WeightsStd', torch.ones_like(TestNetwork.model_stack[0].weight), constraint=torch.distributions.constraints.positive)
    
    Layer1BiasMean = pyro.param('Layer1BiasMean', torch.zeros_like(TestNetwork.model_stack[0].bias))
    Layer1BiasStd = pyro.param('Layer1BiasStd', torch.ones_like(TestNetwork.model_stack[0].bias), constraint=torch.distributions.constraints.positive)
    
    Layer2WeightsMean = pyro.param('Layer2WeightsMean', torch.zeros_like(TestNetwork.model_stack[2].weight))
    Layer2WeightsStd = pyro.param('Layer2WeightsStd', torch.ones_like(TestNetwork.model_stack[2].weight), constraint=torch.distributions.constraints.positive)
    
    Layer2BiasMean = pyro.param('Layer2BiasMean', torch.zeros_like(TestNetwork.model_stack[2].bias))
    Layer2BiasStd = pyro.param('Layer2BiasStd', torch.ones_like(TestNetwork.model_stack[2].bias), constraint=torch.distributions.constraints.positive)
    
    pyro.sample('Layer1Weights', dist.Normal(Layer1WeightsMean, Layer1WeightsStd).independent())
    pyro.sample('Layer1Bias', dist.Normal(Layer1BiasMean, Layer1BiasStd).independent())
    pyro.sample('Layer2Weights', dist.Normal(Layer2WeightsMean, Layer2WeightsStd).independent())
    pyro.sample('Layer2Bias', dist.Normal(Layer2BiasMean, Layer2BiasStd).independent())

guide = pyro.infer.autoguide.AutoDiagonalNormal(LeModel)
#guide = CustomGuide
adam = pyro.optim.Adam({"lr": 0.005})
svi = SVI(LeModel, guide=guide, optim=adam, loss=Trace_ELBO(retain_graph=True))

pyro.clear_param_store()
'''
for step in range(5000):
    #print("Hello")
    loss = svi.step(train_set, train_labels)
    #print("Step:", step, " Loss:", loss)
    if step % 100 == 0:
        print("Step:", step, " Loss:", loss)
'''
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        # Run one step of SVI (this will update the guide parameters)
        loss = svi.step(x_batch, y_batch)  # SVI step with current batch
        total_loss += loss
        
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
    
    # Average loss over the whole epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")


'''
for image in range(5):
    temp_image = test_set[2, :, :]
    with pyro.plate("samples", 5, dim=-1):
        samples = guide(temp_image)
    print(samples)
'''
#pls_work.forward(train_set[0:128])

#nuts_kernel = pyro.infer.NUTS(pls_work, jit_compile=False)
#mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=50)

#mcmc.run(train_set, train_labels)

#predictive = pyro.infer.Predictive(model=pls_work, posterior_samples=mcmc.get_samples())
#preds = predictive(test_set)
