# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:08:30 2025

@author: charl
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import astropy
import torch
import pyro
from astropy.io import fits
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
import pyro.distributions as dist

#Functionally, makes sure that Pytorch and Pyro have been imported and linked correctly
assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

start=time.time()

#Set RNG seed for reproducibility and clear paramater store to make sure we start fresh!
pyro.set_rng_seed(123)
pyro.clear_param_store()

'''  
***CATALOG FEATURES MASTER LIST***
18 unique features in master list (when grouped with uncertainties)
Corresponds to 1 classification label of the source (training label),
and 17 unique inputs for the neural network (property value (+ uncertainty on some of them))
'''
features_master_list = [
                        "TTYPE5",             #GLAT
                        "TTYPE6",             #GLON
                        "TTYPE7",             #Signif_Avg
                        "TTYPE8", "TTYPE9",   #Flux1000 + unc.
                        "TTYPE10", "TTYPE11", #Energy_Flux100 + unc.
                        "TTYPE12",            #Spectrum Type
                        "TTYPE13", "TTYPE14", #PL_Index + unc.
                        "TTYPE15",            #Pivot_Energy
                        "TTYPE16", "TTYPE17", #LP_Index + unc.
                        "TTYPE18", "TTYPE19", #LP_beta + unc.
                        "TTYPE20",            #Flags, ***511 have flags that should be taken into consideration***
                        "TTYPE31",            #SED_class
                        "TTYPE36",            #nu_syn, ***777 are missing data***
                        "TTYPE37",            #nuFnu_syn, ***777 are missing data***
                        "TTYPE38",            #Variability_Index
                        "TTYPE39", "TTYPE40", #Frac_Variability + unc. ***874 are missing data (i.e: value 0 for frac_var, value 10.0 for unc.)***
                        "TTYPE41",             #Highest_energy, ***1286 are missing data***
                        "TTYPE21"            #Classification - DO NOT USE TO TRAIN NETWORK!!!
                        ]

def FeatureDisplay(hdu_table):
    '''
    Displays some basic info about each feature group in the input HDU table 
    
    Inputs: HDU table to be described
    Outputs: Feature Name
             Number of elements
             Number of non-zero elements
             Min. and max. values of each feature array
             (all printed into console)
    '''
    header_array = hdu_table.header
    data_array = hdu_table.data
    
    for x in features_master_list:
        feature_header = header_array[x]
        feature_data = data_array[feature_header]
        print("#####Feature:#####", feature_header, "\nNum. elements:", len(feature_data), "\nNum. non-zero:", np.count_nonzero(feature_data))
        try: print(np.min(feature_data), np.max(feature_data))
        except: print("Not correct data type for min. and max. values")
        
def MissingDataFiltering(hdu_table, filter_bad_sources=True):
    '''
    Filters out sources with missing synchrotron and/or highest energy data
    Also filters out sources with any associated flag *LIKELY ONLY TEMPORARY*
    
    Inputs: HDU table
            *Optional* Boolean depending on whether to filter bad sources or not (default True)
    Returns: HDU headers
             HDU data as separate arrays (NOT the full combined HDU table)
             
    ***After filtering all of these features, only 1263 sources left out of 2630 for 4LAC-DR3***
    '''
    problematic_features = ["TTYPE36", "TTYPE37", "TTYPE39", "TTYPE41"]
    #Only includes the Frac_Var ("TTYPE39") and not its uncertainty ("TTYPE40")
    #All values of 0 for Frac_Var are "missing data", and associated with an unc. of 10.0
    #So no need to filter for unc. too!
    
    header_array = hdu_table.header
    data_array = hdu_table.data
    
    if filter_bad_sources == True:
        temp_filtered_table = data_array
        #print("Starting length: ", len(temp_filtered_table))
        for x in problematic_features:
            feature_header = header_array[x]
            temp_filtered_table = temp_filtered_table[(temp_filtered_table[feature_header] != 0) & (temp_filtered_table[feature_header] != -np.inf)]
            #print(feature_header, " filtered, new length: ", len(temp_filtered_table))
    
        flag_feature = header_array["TTYPE20"]
        temp_filtered_table = temp_filtered_table[(temp_filtered_table[flag_feature] == np.int16(0))]
        #print("Flags filtered, new length: ", len(temp_filtered_table))
        
        data_array = temp_filtered_table

    return header_array, data_array

class FeatureDataset(Dataset):
    '''
    Packages the raw data and class tensors into a Dataset object
    Compatible with DataLoaders, making them easier to batch and pass into the neural network
    
    __init__: Initialises the dataset (assigning the data, classes and any desired transform)
    __len__: Returns the length of the feature tensor
    __getitem__: Returns a single source's data and its correct classification      
    '''
    def __init__(self, feature_tensor, class_tensor, transform=None):
        self.feature_tensor = feature_tensor
        self.class_tensor = class_tensor
        self.transform = transform

    def __len__(self):
        return len(self.feature_tensor)

    def __getitem__(self, idx):
        source_features = self.feature_tensor[idx]
        source_features = source_features.unsqueeze(0)

        source_class = self.class_tensor[idx]
        
        return source_features, source_class
    
def InitialiseDataLoaders(feature_array, class_array, batch_size=64, shuffle=True):
    '''
    Package the input arrays into Pytorch Dataset and DataLoader objects
    Inputs: 2D array for the network's input features
            1D array for the actual source classifications
            *Optional* Integer for batch size to use for DataLoader
            *Optional* Boolean for whether the DataLoader should shuffle the input orders
    Outputs: Tensor for the input features
             Tensor for the actual classifications
             DataLoader object containing the input features and classifications
    '''
    feature_tensor = torch.from_numpy(feature_array)
    class_tensor = torch.from_numpy(class_array)
    
    dataset = FeatureDataset(feature_tensor, class_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return feature_tensor, class_tensor, dataloader

def ClassificationFiltering(input_master_array):
    '''
    Filters the master data array for sources that are not of the type BL Lac or FSRQ
    Stores them all as lower-case string values
    
    Inputs: Master data array of all source data
    Outputs: Master data array with the source info. properly formatted
    '''
    class_feature = "CLASS"
    filtered_array = input_master_array
    
    #print(filtered_array["CLASS"][0:50])
    
    for i in range(len(filtered_array)):
        class_temp = filtered_array[class_feature][i]
        class_temp = class_temp.lower()
        
        if (class_temp != 'fsrq') and (class_temp != 'bll'):
            class_temp = 'inval'
            
        filtered_array[class_feature][i] = class_temp
        
    filtered_array = filtered_array[filtered_array[class_feature] != 'inval']
    #print(filtered_array["CLASS"][0:50])
    
    return filtered_array

def PowerLawFormatting(input_master_array, exclude_super_exp_cutoff=False):
    '''
    Converts the different power-law entries into unique identifiers:
        0 - PowerLaw
        1 - LogParabola
        2 - PLSuperExpCutoff4
    Also includes the option to filter out PLSuperExpCutoff4 types
    (Only 6 of them in the whole catalog, so could be unnecessary consideration for network...
     However, all 6 are strong FSRQ/BL Lac detections!)
    
    Inputs: Master data array of all source data
            *Optional* Boolean depending on whether to filter out PLSuperExpCutoff4
    Outputs: Master data array with the PowerLaw column properly formatted
    '''
    PL_feature = "SpectrumType"
    filtered_array = input_master_array
    
    for i in range(len(filtered_array)):
        PL_temp = filtered_array[PL_feature][i]
        
        if PL_temp == 'PowerLaw':
            PL_temp = 0
        elif PL_temp == 'LogParabola':
            PL_temp = 1
        elif PL_temp == 'PLSuperExpCutoff4':
            PL_temp = 2
        else:
            print("Something went wrong in PL formatting!")
            
        filtered_array[PL_feature][i] = int(PL_temp)

    if exclude_super_exp_cutoff:
        filtered_array = filtered_array[filtered_array[PL_feature] != 2]
        
    return filtered_array

def SEDClassFormatting(input_master_array):
    '''
    Converts the different Spectral Energy Distribution (SED) class entries into unique identifiers:
        0 - LSP
        1 - ISP
        2 - HSP
    
    Inputs: Master data array of all source data
    Outputs: Master data array with the SED class column properly formatted
    '''
    SED_feature = "SED_class"
    filtered_array = input_master_array
    
    for i in range(len(filtered_array)):
        SED_temp = filtered_array[SED_feature][i]
        
        if SED_temp == 'LSP':
            SED_temp = 0
            
        elif SED_temp == 'ISP':
            SED_temp = 1
            
        elif SED_temp == 'HSP':
            SED_temp = 2
            
        else:
            print("Something went wrong in SED formatting!")
            
        filtered_array[SED_feature][i] = int(SED_temp)
    
    return filtered_array

class ClassificationNeuralNetwork(PyroModule[nn.Module]):
    '''
    Defines the Bayesian Neural Network that we use for the classification of BCUs
    Declared as a PyroModule mixin class of nn.Module
    
    __init__: Initialises the network with the network layer sizes and declares the Pyro objects to give the Bayesian behaviour
    forward: Called whenever the network is 'run', we apply the network transformations to the inputs in order and return the outputs
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        #Runs the initialisation of the network's PyTorch superclass (nn.Module)
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        #Initialises the network layers as PyroModules
        self.Layer1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.Layer2 = PyroModule[nn.Linear](hidden_dim, output_dim)
        
        
        #Declare parameters for each network weight as Pyro parameters, mean and std. of a normal distribution
        #########Log stds. declared since it helps with training#########
        self.L1WMean = PyroParam(torch.zeros_like(self.Layer1.weight))
        self.L1WLogStd = PyroParam(-torch.ones_like(self.Layer1.weight))
        self.L1WStd = torch.exp(self.L1WLogStd)

        self.L1BMean = PyroParam(torch.zeros_like(self.Layer1.bias))
        self.L1BLogStd = PyroParam(-torch.ones_like(self.Layer1.bias))
        self.L1BStd = torch.exp(self.L1BLogStd)


        self.L2WMean = PyroParam(torch.zeros_like(self.Layer2.weight))
        self.L2WLogStd = PyroParam(-torch.ones_like(self.Layer2.weight))
        self.L2WStd = torch.exp(self.L2WLogStd)

        self.L2BMean = PyroParam(torch.zeros_like(self.Layer2.bias))
        self.L2BLogStd = PyroParam(-torch.ones_like(self.Layer2.bias))
        self.L2BStd = torch.exp(self.L2BLogStd)


        #Declare layer weights and biases as Pyro samples from normal distribution based on previously-declared Pyro parameters
        self.Layer1.weight = PyroSample(dist.Normal(self.L1WMean, self.L1WStd).independent(2))
        self.Layer1.bias = PyroSample(dist.Normal(self.L1BMean, self.L1BStd).independent(1))

        self.Layer2.weight = PyroSample(dist.Normal(self.L2WMean, self.L2WStd).independent(2))
        self.Layer2.bias = PyroSample(dist.Normal(self.L2BMean, self.L2BStd).independent(1))

    def forward(self, input_data):
        #Simply runs through the network stack and returns the outputs
        
        #print(self.Layer2.bias)
        input_data = self.flatten(input_data)
        hidden_data = nn.functional.relu(self.Layer1(input_data))
        output_data = self.Layer2(hidden_data)
        return output_data

#Initialise the neural network instance for the model and guide
input_nodes = 23
hidden_nodes = 10
output_nodes = 2
neural_network = ClassificationNeuralNetwork(input_nodes, hidden_nodes, output_nodes)

def ModelFunc(input_features, correct_labels=None, anneal_factor=1.0):
    '''
    Runs the model on the given input feature tensor, then sampling a classification from these logits
    
    Inputs: Tensor of the input features to be passed into the neural network, 
            *Optional* Tensor of the correct classifications
    Returns: Tensor of the actual correct labels itself (if passed into the function)
             Tensor of the sampled values from the calculated logits if not
    '''
    logits = neural_network(input_features)
    with pyro.plate("results", logits.shape[0]):
        return pyro.sample("obs", dist.Categorical(logits=logits), obs=correct_labels)    
    
def CustomGuide(input_features, correct_labels, anneal_factor=1.0):
    '''
    Samples the weights of the neural network from the global parameters
    The poutine.scale() applies the KL-annealing factor, allowing some burn-in for the parameters
    Necessary since the priors being declared as standard normals isn't particularly accurate
    
    Inputs: Tensor of the inputs features
            Tensor of the correct classification labels, optional KL-annealing strength factor
    '''
    
    #Scales the log_std size (e.g: a value of -2 means stds of e^-2 = 0.136...)
    std_scale = -2
    
    #Local copies of the global weight distribution parameters
    L1W_Mean = pyro.param('L1WMean', torch.zeros_like(neural_network.Layer1.weight))
    L1W_LogStd = pyro.param('L1WLogStd', std_scale*torch.ones_like(neural_network.Layer1.weight))
    L1W_Std = torch.exp(L1W_LogStd)
    
    L1B_Mean = pyro.param('L1BMean', torch.zeros_like(neural_network.Layer1.bias))
    L1B_LogStd = pyro.param('L1BLogStd', std_scale*torch.ones_like(neural_network.Layer1.bias))
    L1B_Std = torch.exp(L1B_LogStd)
        
    L2W_Mean = pyro.param('L2WMean', torch.zeros_like(neural_network.Layer2.weight))
    L2W_LogStd = pyro.param('L2WLogStd', std_scale*torch.ones_like(neural_network.Layer2.weight))
    L2W_Std = torch.exp(L2W_LogStd)
    
    L2B_Mean = pyro.param('L2BMean', torch.zeros_like(neural_network.Layer2.bias))
    L2B_LogStd = pyro.param('L2BLogStd', std_scale*torch.ones_like(neural_network.Layer2.bias))
    L2B_Std = torch.exp(L2B_LogStd)
    
    #New weights and biases are sampled with these local parameters
    with pyro.poutine.scale(None, anneal_factor):
        pyro.sample('Layer1.weight', dist.Normal(L1W_Mean, L1W_Std).independent(2))
        pyro.sample('Layer1.bias', dist.Normal(L1B_Mean, L1B_Std).independent(1))
        pyro.sample('Layer2.weight', dist.Normal(L2W_Mean, L2W_Std).independent(2))
        pyro.sample('Layer2.bias', dist.Normal(L2B_Mean, L2B_Std).independent(1))


##### MAIN CODE #####

#Import catalog and filter out missing data
fits_file = fits.open("table-4LAC-DR3-h.fits")
hdu_table = fits_file[1]
master_headers_array, master_data_array = MissingDataFiltering(hdu_table)

#Format the string-based columns into network-readable numerical values
master_data_array = PowerLawFormatting(master_data_array)
master_data_array = SEDClassFormatting(master_data_array)

## TRAINING DATA CREATION ##
#Filter out non-relevant classifications for training set (FSRQs and BL Lacs)
temp_data_array = ClassificationFiltering(master_data_array)

#Split master data into network training data and classification "correct answers"
train_data_array = np.zeros((len(temp_data_array), len(features_master_list)-1), dtype=np.float32)
train_class_array = np.array(len(temp_data_array), dtype=str)

for i in range(len(features_master_list)):
    feature = master_headers_array[features_master_list[i]]
    
    if i==23: #Splits off the classification array
        train_class_array = temp_data_array[feature]
        
        #Convert class strings to numerical values - easier to compare predictions to actual values!
        train_class_array[train_class_array == 'fsrq'] = 1
        train_class_array[train_class_array == 'bll'] = 0
        train_class_array = np.asarray(train_class_array, dtype=int)
        
    else:
        train_data_array[:, i] = temp_data_array[feature]
    
#Log transformations of some of the columns since they have values over many orders of magnitude
indices_to_log = [3, 4, 5, 6, 9, 12, 14, 17, 18, 19]
for index in indices_to_log:
    train_data_array[:, index] = np.log(train_data_array[:, index])
       
#Further split into training set, testing set, and validation set - 80% training set, 10% each validation and test sets
split = int(round(0.8 * len(train_data_array)))
test_data_array, test_class_array = train_data_array[split:], train_class_array[split:]
train_data_array, train_class_array = train_data_array[0:split], train_class_array[0:split]

split = int(round(0.5 * len(test_data_array)))
val_data_array, val_class_array = test_data_array[split:], test_class_array[split:]
test_data_array, test_class_array = test_data_array[0:split], test_class_array[0:split]

#Convert data arrays into Torch tensors and initialise DataLoaders 
train_data_tensor, train_class_tensor, train_dataloader = InitialiseDataLoaders(train_data_array, train_class_array, batch_size=16)
val_data_tensor, val_class_tensor, val_dataloader = InitialiseDataLoaders(val_data_array, val_class_array)
test_data_tensor, test_class_tensor, test_dataloader = InitialiseDataLoaders(test_data_array, test_class_array)


## NEURAL NETWORK TRAINING PROCESS ##
def Sigmoid(epoch, total_epochs, ramp_factor=1/2):
    '''
    Applies a sigmoid function to the current epoch to find the KL annealing factor
    Starts small, quickly ramps up midway through training, high value towards the end of training
    Inputs: Int for the current epoch
            Int for the total number of epochs
            *Optional* Float to scale the effective midpoint; changing this changes how early the KL annealing "ramp-up" happens
    Outputs: Float for the KL annealing factor to use for the current epoch
    *Note that ramp_factor and the "-0.1" in the exponential are tuneable hyperparameters for the profile of the sigmoid function*
    '''
    sigmoid_midpoint = ramp_factor * total_epochs
    return 1/(1 + np.exp(-0.1 * (epoch-sigmoid_midpoint)))

def MCMCMethod():
    num_samples=500
    nuts_kernel = pyro.infer.NUTS(ModelFunc, jit_compile=False)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_samples)
    mcmc.run(train_data_tensor, train_class_tensor)
    
    for i in range(20):
        predictive = pyro.infer.Predictive(model=ModelFunc, posterior_samples=mcmc.get_samples())
        preds = predictive(val_data_tensor)
        
        correct_number=0
        for x in range(num_samples):
            correct_number += (preds['obs'][x].cpu().numpy() == val_class_tensor.cpu().numpy())
        
        #print(correct_number)
        accuracy_rate = 100*(correct_number/num_samples)
        print(np.mean(accuracy_rate))

def SVIMethod():
    #Run over a number of epochs (number of times to iterate through the dataset)
    num_epochs = 5000
    temp_decay = 0.01**(1/(54*num_epochs)) #Final learning rate is 0.01*initial learning rate
    print(temp_decay)
    
    #Define the network guide, network optimiser, and SVI training method
    guide = CustomGuide
    adam = pyro.optim.ClippedAdam({'lr': 5e-4, 'lrd': temp_decay})
    svi = SVI(ModelFunc, guide=guide, optim=adam, loss=Trace_ELBO(retain_graph=True))
    
    last_10_accuracies = np.zeros(10)+50
    losses_array = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        anneal_factor = 1#Sigmoid(epoch, num_epochs)
        #print(f"Epoch {epoch+1}, KL Annealing Factor: {anneal_factor}")
        
        #Iterates through entire dataset, doing a SVI step after each batch to improve efficiency
        neural_network.train()
        for batch_index, (current_train_data, current_train_class) in enumerate(train_dataloader):
            loss = svi.step(current_train_data, current_train_class, anneal_factor=anneal_factor)
            epoch_loss += loss
            
            #if batch_index % 10 == 0:
            #    print(f"Epoch: {epoch+1}; Batch: {batch_index}, Loss: {epoch_loss:.4f}")
        
        #Output average loss value over the whole epoch
        average_loss = epoch_loss/len(train_dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1} completed! Average loss: {average_loss:.4f}") 
            print("Average log_std for 1st layer weights: ", np.mean(pyro.get_param_store()['L1WLogStd'].detach().cpu().numpy()))    
            #print("Average log_std for 2nd layer weights: ", np.mean(pyro.get_param_store()['L2WLogStd'].detach().cpu().numpy()))
        losses_array[epoch] = average_loss
        
        #Start the validation to test model accuracy after an epoch of training
        neural_network.eval()
        current_val_data, current_val_class = next(iter(test_dataloader))
        correct_number = np.zeros(current_val_class.shape[0])
        
        #Number of times to sample the predictions for each data point - higher number = less spread, but more computation time!
        sample_number = 200
        for sample in range(sample_number):
            #Sample the posterior weights and run the neural network on these
            guide_trace = pyro.poutine.trace(guide).get_trace(None, None)
            network_replay = pyro.poutine.replay(neural_network, trace=guide_trace)
            
            #Generates predictions from the network - no gradient tracking needed
            with torch.no_grad():
                logits = network_replay(current_val_data)
                probs = nn.functional.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                #if epoch == 9:
                #    print(probs, predictions)
    
            correct_number += (predictions.cpu().numpy() == current_val_class.cpu().numpy())
        
        accuracy_rate = (correct_number/sample_number)*100
        average_accuracy = np.mean(accuracy_rate)
        last_10_accuracies[epoch%10] = average_accuracy
        #print(f"Average accuracy after Epoch {epoch+1}: {average_accuracy:.2f}%")
        if epoch % 10 == 0:
            print(f"Rolling average accuracy (last 10 epochs): {np.mean(last_10_accuracies):.2f}%")

    return losses_array
losses_array = SVIMethod()
#plt.scatter(np.linspace(0, num_epochs, num_epochs+1), losses_array, marker='x', color='black')
finish = time.time()
print("Run time:", finish-start, "seconds")

import os
import time
time.sleep(10)
os.system("shutdown.exe /h")