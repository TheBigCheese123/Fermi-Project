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
from torch.utils.data import Dataset, DataLoader
import pyro.contrib.examples.util

pyro.set_rng_seed(1)
pyro.clear_param_store()

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

class MNISTDataset(Dataset):
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

train_dataset = MNISTDataset(train_set, train_labels)
test_dataset = MNISTDataset(test_set, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

input_dim = 784
hidden_dim = 512
output_dim = 10

#test = pyro.nn.DenseNN(input_dim, [512], param_dims=[output_dim], nonlinearity=nn.Softmax(dim=10))
#test.flatten = nn.Flatten()
#print(test)

class NeuralNetwork(PyroModule[nn.Module]):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()

        self.Layer1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.Layer2 = PyroModule[nn.Linear](hidden_dim, output_dim)
        
        self.Layer1WeightsMean = PyroParam(torch.zeros_like(self.Layer1.weight))
        self.Layer1WeightsStd = PyroParam(torch.ones_like(self.Layer1.weight), constraint=torch.distributions.constraints.positive)

        self.Layer1BiasMean = PyroParam(torch.zeros_like(self.Layer1.bias))
        self.Layer1BiasStd = PyroParam(torch.ones_like(self.Layer1.bias), constraint=torch.distributions.constraints.positive)
        
        self.Layer2WeightsMean = PyroParam(torch.zeros_like(self.Layer2.weight))
        self.Layer2WeightsStd = PyroParam(torch.ones_like(self.Layer2.weight), constraint=torch.distributions.constraints.positive)
        
        self.Layer2BiasMean = PyroParam(torch.zeros_like(self.Layer2.bias))
        self.Layer2BiasStd = PyroParam(torch.ones_like(self.Layer2.bias), constraint=torch.distributions.constraints.positive)
        
        self.Layer1.weight = PyroSample(dist.Normal(self.Layer1WeightsMean, self.Layer1WeightsStd).independent(2))
        #self.fc1.weight = self.Layer1Weights
        
        self.Layer1.bias = PyroSample(dist.Normal(self.Layer1BiasMean, self.Layer1BiasStd).independent(1))
        #self.fc1.bias = self.Layer1Bias
        
        self.Layer2.weight = PyroSample(dist.Normal(self.Layer2WeightsMean, self.Layer2WeightsStd).independent(2))
        #self.fc2.weight = self.Layer2Weights
        
        self.Layer2.bias = PyroSample(dist.Normal(self.Layer2BiasMean, self.Layer2BiasStd).independent(1))
        #self.fc2.bias = self.Layer2Bias
        
    def forward(self, input_data):
        #print(self.model_stack[0].weight)
        
        input_data = self.flatten(input_data)
        hidden_data = torch.nn.functional.leaky_relu(self.Layer1(input_data))
        output_data = self.Layer2(hidden_data)
        return output_data

TestNetwork = pyro.module("net", NeuralNetwork(input_dim, hidden_dim, output_dim))

def LeModel(image_set, image_labels=None):
    #Layer1Weights = pyro.sample('Layer1Weights', dist.Normal(0, 1).expand([input_dim, 512]).to_event(2))
    #Layer1Bias = pyro.sample('Layer1Bias', dist.Normal(0, 1).expand([512]).to_event(1))

    #Layer2Weights = pyro.sample('Layer2Weights', dist.Normal(0, 1).expand([512, output_dim]).to_event(2))
    #Layer2Bias = pyro.sample('Layer2Bias', dist.Normal(0, 1).expand([output_dim]).to_event(1))

    #priors = {'L1W': Layer1Weights, 'L1B': Layer1Bias, 'L2W': Layer2Weights, 'L2B': Layer2Bias}
    
    logits = TestNetwork(image_set)

    with pyro.plate("results", logits.shape[0]):
        return pyro.sample("obs", dist.Categorical(logits=logits), obs=image_labels)

def CustomGuide(image_set, image_labels):
    std_scale = 0.7
    
    Layer1WeightsMean = pyro.param('Layer1WeightsMean', torch.zeros_like(TestNetwork.Layer1.weight))
    Layer1WeightsStd = pyro.param('Layer1WeightsStd', std_scale*torch.ones_like(TestNetwork.Layer1.weight), constraint=torch.distributions.constraints.positive)
    
    Layer1BiasMean = pyro.param('Layer1BiasMean', torch.zeros_like(TestNetwork.Layer1.bias))
    Layer1BiasStd = pyro.param('Layer1BiasStd', std_scale*torch.ones_like(TestNetwork.Layer1.bias), constraint=torch.distributions.constraints.positive)
    
    Layer2WeightsMean = pyro.param('Layer2WeightsMean', torch.zeros_like(TestNetwork.Layer2.weight))
    Layer2WeightsStd = pyro.param('Layer2WeightsStd', std_scale*torch.ones_like(TestNetwork.Layer2.weight), constraint=torch.distributions.constraints.positive)
    
    Layer2BiasMean = pyro.param('Layer2BiasMean', torch.zeros_like(TestNetwork.Layer2.bias))
    Layer2BiasStd = pyro.param('Layer2BiasStd', std_scale*torch.ones_like(TestNetwork.Layer2.bias), constraint=torch.distributions.constraints.positive)
    
    pyro.sample('Layer1.weight', dist.Normal(Layer1WeightsMean, Layer1WeightsStd).independent(2))
    pyro.sample('Layer1.bias', dist.Normal(Layer1BiasMean, Layer1BiasStd).independent(1))
    pyro.sample('Layer2.weight', dist.Normal(Layer2WeightsMean, Layer2WeightsStd).independent(2))
    pyro.sample('Layer2.bias', dist.Normal(Layer2BiasMean, Layer2BiasStd).independent(1))

    #new_network = pyro.module("net", NeuralNetwork)

#guide = pyro.infer.autoguide.AutoNormal(LeModel)
guide = CustomGuide
adam = pyro.optim.Adam({"lr": 0.001, "weight_decay":1e-5})
svi = SVI(LeModel, guide=guide, optim=adam, loss=Trace_ELBO(retain_graph=True))

'''
for step in range(5000):
    #print("Hello")
    loss = svi.step(train_set, train_labels)
    #print("Step:", step, " Loss:", loss)
    if step % 100 == 0:
        print("Step:", step, " Loss:", loss)
'''
num_epochs = 30
for epoch in range(num_epochs):
    total_loss = 0.0
    
    for batch_idx, (temp_train_images, temp_train_labels) in enumerate(train_loader):
        # Run one step of SVI (this will update the guide parameters)
        loss = svi.step(temp_train_images, temp_train_labels)  # SVI step with current batch
        total_loss += loss
        
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
            #print(pyro.get_param_store()['Layer1WeightsMean_unconstrained'][0, 0:8])    
            #print(pyro.get_param_store()['Layer1WeightsStd_unconstrained'][0, 0:8])
    
    # Average loss over the whole epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    temp_test_images, temp_test_labels = next(iter(test_loader))
    
    sample_number = 500
    correct_number = np.zeros(temp_test_images.shape[0])
    
    for i in range(sample_number):
        # Sample posterior weights
        guide_trace = pyro.poutine.trace(guide).get_trace(temp_test_images, None)
        model_replay = pyro.poutine.replay(TestNetwork, trace=guide_trace)
                
        with torch.no_grad():
            logits = model_replay(temp_test_images)
            preds = torch.argmax(logits, dim=-1)
        
        correct_number += (preds.cpu().numpy() == temp_test_labels.cpu().numpy())
        
    accuracy_rate = correct_number/sample_number
    avg_accuracy = np.mean(accuracy_rate)*100
    print(f"Epoch {epoch+1}, Average Accuracy: {avg_accuracy:.4f}%")

#pls_work.forward(train_set[0:128])

#plt.imshow(test_set[4])
#LeModel(test_set[4:5])

#256 nodes:
#Epoch 10, Average Loss: 2135.4696
#Epoch 10, Average Accuracy: 47.3672%

#nuts_kernel = pyro.infer.NUTS(pls_work, jit_compile=False)
#mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=50)

#mcmc.run(train_set, train_labels)

#predictive = pyro.infer.Predictive(model=pls_work, posterior_samples=mcmc.get_samples())
#preds = predictive(test_set)
