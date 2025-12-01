# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:08:30 2025

@author: charl
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

#Set RNG seed for reproducibility and clear paramater store to make sure we start fresh!
pyro.set_rng_seed(123)
pyro.clear_param_store()

'''  
***CATALOG FEATURES MASTER LIST***
18 unique features in master list (when grouped with uncertainties)
Corresponds to 1 classification label of the source (training label),
and 17 unique inputs for the neural network (property value (+ uncertainty on some of them))
'''
features_master_array = [
                        "TTYPE21",            #Class
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
                        "TTYPE41"             #Highest_energy, ***1286 are missing data***
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
            Boolean depending on whether to filter bad sources or not (default True)
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
        self.L1W_Mean = PyroParam(torch.zeros_like(self.Layer1.weight))
        self.L1W_LogStd = PyroParam(torch.zeros_like(self.Layer1.weight))
        self.L1W_Std = torch.exp(self.L1W_LogStd)

        self.L1B_Mean = PyroParam(torch.zeros_like(self.Layer1.bias))
        self.L1B_LogStd = PyroParam(torch.zeros_like(self.Layer1.bias))
        self.L1B_Std = torch.exp(self.L1B_LogStd)


        self.L2W_Mean = PyroParam(torch.zeros_like(self.Layer2.weight))
        self.L2W_LogStd = PyroParam(torch.zeros_like(self.Layer2.weight))
        self.L2W_Std = torch.exp(self.L2W_LogStd)

        self.L2B_Mean = PyroParam(torch.zeros_like(self.Layer2.bias))
        self.L2B_LogStd = PyroParam(torch.zeros_like(self.Layer2.bias))
        self.L2B_Std = torch.exp(self.L2B_LogStd)


        #Declare layer weights and biases as Pyro samples from normal distribution based on previously-declared Pyro parameters
        self.Layer1.weight = PyroSample(dist.Normal(self.L1W_Mean, self.L1W_Std).independent(2))
        self.Layer1.bias = PyroSample(dist.Normal(self.L1B_Mean, self.L1B_Std).independent(1))

        self.Layer2.weight = PyroSample(dist.Normal(self.L2W_Mean, self.L2W_Std).independent(2))
        self.Layer2.bias = PyroSample(dist.Normal(self.L2B_Mean, self.L2B_Std).independent(1))

    def forward(self, input_data):
        #Simply runs through the network stack and returns the outputs
        input_data = self.flatten(input_data)
        hidden_data = nn.functional.relu(self.Layer1(input_data))
        output_data = self.Layer2(hidden_data)
        return output_data

#Initialise the neural network instance for the model and guide
input_nodes = 17
hidden_nodes = 64
output_nodes = 2
neural_network = ClassificationNeuralNetwork(input_nodes, hidden_nodes, output_nodes)

def Model(input_features, correct_labels=None):
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
    
def Guide(inputs_features, correct_labels, annealing_factor=1.0):
    '''
    Samples the weights of the neural network from the global parameters
    The poutine.scale() applies the KL-annealing factor, allowing some burn-in for the parameters
    Necessary since the priors being declared as standard normals isn't particularly accurate
    
    Inputs: Tensor of the inputs features
            Tensor of the correct classification labels, optional KL-annealing strength factor
    '''
    
    #Local copies of the global weight distribution parameters
    L1W_Mean = pyro.param('L1W_Mean', torch.zeros_like(neural_network.Layer1.weight))
    L1W_LogStd = pyro.param('L1W_LogStd', torch.zeros_like(neural_network.Layer1.weight))
    L1W_Std = torch.exp(L1W_LogStd)
    
    L1B_Mean = pyro.param('L1B_Mean', torch.zeros_like(neural_network.Layer1.bias))
    L1B_LogStd = pyro.param('L1B_Mean', torch.zeros_like(neural_network.Layer1.bias))
    L1B_Std = torch.exp(L1B_LogStd)
        
    L2W_Mean = pyro.param('L2W_Mean', torch.zeros_like(neural_network.Layer2.weight))
    L2W_LogStd = pyro.param('L2W_LogStd', torch.zeros_like(neural_network.Layer2.weight))
    L2W_Std = torch.exp(L2W_LogStd)
    
    L2B_Mean = pyro.param('L2B_Mean', torch.zeros_like(neural_network.Layer2.bias))
    L2B_LogStd = pyro.param('L2B_Mean', torch.zeros_like(neural_network.Layer2.bias))
    L2B_Std = torch.exp(L2B_LogStd)
    
    #New weights and biases are sampled with these local parameters
    with pyro.poutine.scale(None, anneal_factor):
        pyro.sample('Layer1.weight', dist.Normal(L1W_Mean, L1W_Std).independent(2))
        pyro.sample('Layer1.bias', dist.Normal(L1B_Mean, L1B_Std).independent(1))
        pyro.sample('Layer2.weight', dist.Normal(L2W_Mean, L2W_Std).independent(2))
        pyro.sample('Layer2.bias', dist.Normal(L2B_Mean, L2B_Std).independent(1))

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
        
        return source_data, source_class
    
def ClassificationFiltering(input_master_array):
    '''
    Filters the master data array for sources that are not of the type BL Lac or FSRQ
    Then converts these class strings into numerical values that the 
    
    Inputs: Master data array of all source data
    Outputs: Integer array of equivalent classifications
    '''
    class_feature = "CLASS"
    filtered_array = input_master_array
    
    #print(filtered_array["CLASS"][0:50])
    
    for i in range(len(filtered_array)):
        class_temp = filtered_array[class_feature][i]
        class_temp = class_temp.lower()
        
        if (class_temp != 'fsrq') and (class_temp != 'bll') and (class_temp != 'bcu'):
            class_temp = 'inval'
            
        filtered_array[class_feature][i] = class_temp
        
    filtered_array = filtered_array[filtered_array[class_feature] != 'inval']
    #print(filtered_array["CLASS"][0:50])
    
    return filtered_array
        

##### MAIN CODE #####
fits_file = fits.open("table-4LAC-DR3-h.fits")
hdu_table = fits_file[1]
master_headers_array, master_data_array = MissingDataFiltering(hdu_table)

print(np.unique(master_data_array["CLASS"], return_counts=True))
master_data_array = ClassificationFiltering(master_data_array)
print(np.unique(master_data_array["CLASS"], return_counts=True))

source_data_array = np.zeros((len(master_data_array), len(features_master_array)-1))
source_classifications_array = np.array(len(master_data_array), dtype=str)

'''
for i in range(len(features_master_array)):
    feature = features_master_array[i]
    
    if i==0:
        source_classifications_array[:] = ClassificationFormatting(master_data_array)
    else:
        source_data_array[i-1, :] =  master_data_array[master_headers_array[feature]]
    
    print(source_data_array.shape)
    print(source_classifications_array.shape)
'''
    
#ClassificationFiltering(master_data_array)

#for i in range(len(master_network_headers[0])):
#    print(master_network_headers[i])
    #master_network_data = master_data_array



#master_network_dataset = FeatureDataset()

