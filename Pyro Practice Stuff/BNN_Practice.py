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
import time

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
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = image.unsqueeze(0)

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
hidden_dim = 1024
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
        self.Layer1WeightsLogStd = PyroParam(torch.zeros_like(self.Layer1.weight))
        self.L1WS = torch.exp(self.Layer1WeightsLogStd)

        self.Layer1BiasMean = PyroParam(torch.zeros_like(self.Layer1.bias))
        self.Layer1BiasLogStd = PyroParam(torch.zeros_like(self.Layer1.bias))
        self.L1BS = torch.exp(self.Layer1BiasLogStd)

        self.Layer2WeightsMean = PyroParam(torch.zeros_like(self.Layer2.weight))
        self.Layer2WeightsLogStd = PyroParam(torch.zeros_like(self.Layer2.weight))
        self.L2WS = torch.exp(self.Layer2WeightsLogStd)

        self.Layer2BiasMean = PyroParam(torch.zeros_like(self.Layer2.bias))
        self.Layer2BiasLogStd = PyroParam(torch.zeros_like(self.Layer2.bias))
        self.L2BS = torch.exp(self.Layer2BiasLogStd)

        self.Layer1.weight = PyroSample(dist.Normal(self.Layer1WeightsMean, self.L1WS).independent(2))
        self.Layer1.bias = PyroSample(dist.Normal(self.Layer1BiasMean, self.L1BS).independent(1))
        
        self.Layer2.weight = PyroSample(dist.Normal(self.Layer2WeightsMean, self.L2WS).independent(2))
        self.Layer2.bias = PyroSample(dist.Normal(self.Layer2BiasMean, self.L2BS).independent(1))
        
    def forward(self, input_data):
        #print("Hi", self.Layer1.weight)
        
        input_data = self.flatten(input_data)
        hidden_data = nn.functional.relu(self.Layer1(input_data))
        output_data = self.Layer2(hidden_data)
        return output_data

#Loads in parameters from external file - saves retraining for every new kernel instance!
pyro.get_param_store().load("parameters")

TestNetwork = NeuralNetwork(input_dim, hidden_dim, output_dim)

def LeModel(image_set, image_labels=None, anneal_factor=1.0):
    logits = TestNetwork(image_set)
    with pyro.plate("results", logits.shape[0]):
        return pyro.sample("obs", dist.Categorical(logits=logits), obs=image_labels)

def CustomGuide(image_set, image_labels, anneal_factor=1.0):
    std_scale = 0.5
    
    Layer1WeightsMean = pyro.param('Layer1WeightsMean', torch.zeros_like(TestNetwork.Layer1.weight))
    Layer1WeightsLogStd = pyro.param('Layer1WeightsLogStd', std_scale*torch.zeros_like(TestNetwork.Layer1.weight))
    L1WS = torch.exp(Layer1WeightsLogStd)
    
    Layer1BiasMean = pyro.param('Layer1BiasMean', torch.zeros_like(TestNetwork.Layer1.bias))
    Layer1BiasLogStd = pyro.param('Layer1BiasLogStd', std_scale*torch.zeros_like(TestNetwork.Layer1.bias))
    L1BS = torch.exp(Layer1BiasLogStd)

    Layer2WeightsMean = pyro.param('Layer2WeightsMean', torch.zeros_like(TestNetwork.Layer2.weight))
    Layer2WeightsLogStd = pyro.param('Layer2WeightsLogStd', std_scale*torch.zeros_like(TestNetwork.Layer2.weight))
    L2WS = torch.exp(Layer2WeightsLogStd)

    Layer2BiasMean = pyro.param('Layer2BiasMean', torch.zeros_like(TestNetwork.Layer2.bias))
    Layer2BiasLogStd = pyro.param('Layer2BiasLogStd', std_scale*torch.zeros_like(TestNetwork.Layer2.bias))
    L2BS = torch.exp(Layer2BiasLogStd)
    
    with pyro.poutine.scale(None, anneal_factor):
        pyro.sample('Layer1.weight', dist.Normal(Layer1WeightsMean, L1WS).independent(2))
        pyro.sample('Layer1.bias', dist.Normal(Layer1BiasMean, L1BS).independent(1))
        pyro.sample('Layer2.weight', dist.Normal(Layer2WeightsMean, L2WS).independent(2))
        pyro.sample('Layer2.bias', dist.Normal(Layer2BiasMean, L2BS).independent(1))

#guide = pyro.infer.autoguide.AutoNormal(LeModel)
guide = CustomGuide
adam = pyro.optim.ClippedAdam({"lr": 0.005})
svi = SVI(LeModel, guide=guide, optim=adam, loss=Trace_ELBO(retain_graph=True))

'''
for step in range(5000):
    #print("Hello")
    loss = svi.step(train_set, train_labels)
    #print("Step:", step, " Loss:", loss)
    if step % 100 == 0:
        print("Step:", step, " Loss:", loss)
'''

def Sigmoid(epoch, total_epochs):
    #Effective midpoint, changing this changes how early the KL annealing "ramp-up" happens
    sigmoid_midpoint = total_epochs/3
    return 1/(1 + np.exp(-0.2 * (epoch-sigmoid_midpoint)))

num_epochs = 1

for epoch in range(num_epochs):
    total_loss = 0.0
    
    anneal_factor = Sigmoid(epoch, num_epochs)
    print(f"Epoch {epoch+1}, KL Annealing Factor: {anneal_factor}")

    for batch_idx, (temp_train_images, temp_train_labels) in enumerate(train_loader):
        
        #start = time.time()
        loss = svi.step(temp_train_images, temp_train_labels, anneal_factor=anneal_factor)  # SVI step with current batch
        total_loss += loss
        #end = time.time()
        #print(end-start)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
            print("Average log_std mag. for 1st layer weights: ", np.mean(np.abs(pyro.get_param_store()['Layer1WeightsLogStd'].detach().cpu().numpy())))    
            print("Average log_std mag. for 2nd layer weights: ", np.mean(np.abs(pyro.get_param_store()['Layer1BiasLogStd'].detach().cpu().numpy())))
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    temp_test_images, temp_test_labels = next(iter(test_loader))
    sample_number = 100
    correct_number = np.zeros(temp_test_images.shape[0])
    
    for i in range(sample_number):
        #Sample posterior weights
        guide_trace = pyro.poutine.trace(guide).get_trace(None, None)
        model_replay = pyro.poutine.replay(TestNetwork, trace=guide_trace)
                
        with torch.no_grad():
            logits = model_replay(temp_test_images)
            probs = nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            #print(preds, temp_test_labels)
        
        correct_number += (preds.cpu().numpy() == temp_test_labels.cpu().numpy())
        
    accuracy_rate = correct_number/sample_number
    avg_accuracy = np.mean(accuracy_rate)*100
    print(f"Epoch {epoch+1}, Average Accuracy: {avg_accuracy:.4f}%")

###Final accuracy test###
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True)

temp_test_images, temp_test_labels = next(iter(test_loader))
sample_number = 500
correct_number = np.zeros(temp_test_images.shape[0])

print("Training has ended! Starting the final testing run now...")
for i in range(sample_number):
    if i%100 == 0:
        print(f"{i} / {sample_number} samples done!")
    #Sample posterior weights
    guide_trace = pyro.poutine.trace(guide).get_trace(None, None)
    model_replay = pyro.poutine.replay(TestNetwork, trace=guide_trace)
    
    with torch.no_grad():
        logits = model_replay(temp_test_images)
        probs = nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    
    correct_number += (preds.cpu().numpy() == temp_test_labels.cpu().numpy())

accuracy_percentages = (correct_number/sample_number)*100 #For each tested image
avg_accuracy = np.mean(accuracy_percentages)
print(f"End of Training Test, Average Accuracy: {avg_accuracy:.4f}%")

'''
fig = plt.figure(figsize=(10,6))
ax = fig.gca()
ax.hist(accuracy_percentages, bins=20, label="Accuracy Bins")

percentile_20 = np.percentile(accuracy_percentages, 20) #80% of image accuracies are above this value
percentile_50 = np.percentile(accuracy_percentages, 50) #50% of image accuracies are above this value

ax.plot([percentile_20, percentile_20], [0, 1750], lw=2, ls='--', color='black', label=f'20th percentile ({percentile_20: .1f}%)')
ax.plot([percentile_50, percentile_50], [0, 1750], lw=2, ls='--', color='red', label=f'50th percentile ({percentile_50: .1f}%)')

fig.suptitle(f"Distribution of prediction accuracy rates on all 10000 test images, {sample_number} sampling runs")
ax.set_xlabel("Correct Prediction Rate / %")
ax.set_ylabel("Number of Images")
plt.legend()
plt.show()
'''

file_name = input("Parameters file name (type \"n\" to skip saving): ")
if file_name != ("n"):
    pyro.get_param_store().save(file_name)
    print("Parameters saved as \"{file_name}\"")
else:
    print("Parameters not saved")
    

def ImagePredictionsPlot(image_number):
    sample_number = 20
    image_predictions = np.zeros(sample_number)
    
    #image_number=np.random.randint(0, 128)
    #image_number=78
    digits = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int)
    
    for sample in range(sample_number):
        guide_trace = pyro.poutine.trace(guide).get_trace(None, None)
        model_replay = pyro.poutine.replay(TestNetwork, trace=guide_trace)
    
        with torch.no_grad():
            logits = model_replay(temp_test_images[image_number])
            probs = nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
        image_predictions[sample] = preds.cpu().numpy()
            
    grouped_preds = np.unique(image_predictions, return_counts=True)
    
    for x in range(len(digits[0])):
        digit = digits[0, x]
        temp = np.where(grouped_preds[0] == digit)
        if temp[0].size > 0:
            digits[1, x] = grouped_preds[1][temp]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(temp_test_images[image_number].cpu().numpy()[0, :, :])
    ax2.bar(digits[0], digits[1])
    ax1.set_title(f"Image {image_number} in test set - actual label {temp_test_labels[image_number]}")
    ax2.set_xlabel("Predicted digit")
    ax2.set_ylabel("Frequency")
    fig.suptitle(f"Test image and frequency of predictions over {sample_number} samples")
    plt.show()
    plt.pause(3)
    plt.close()

for i in range(0, 127):
    ImagePredictionsPlot(i)

#256 nodes:
#Epoch 10, Average Loss: 2135.4696
#Epoch 10, Average Accuracy: 47.3672%

#pls_work.forward(train_set[0:128])
#plt.imshow(test_set[4])
#LeModel(test_set[4:5])

#nuts_kernel = pyro.infer.NUTS(pls_work, jit_compile=False)
#mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=50)

#mcmc.run(train_set, train_labels)

#predictive = pyro.infer.Predictive(model=pls_work, posterior_samples=mcmc.get_samples())
#preds = predictive(test_set)
