# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:08:30 2025

@author: charl
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import astropy
import torch
import pyro
import arviz as az
from astropy.io import fits
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
import pyro.distributions as dist
import sklearn.metrics

#Functionally, makes sure that Pytorch and Pyro have been imported and linked correctly
assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

start=time.time()

#Set RNG seed for reproducibility and clear paramater store to make sure we start fresh!
rng_seed = 1234
pyro.set_rng_seed(rng_seed)
pyro.clear_param_store()

print("Pyro RNG seed:", rng_seed)

'''  
***CATALOG FEATURES MASTER LIST***
18 unique features in master list (when grouped with uncertainties; 24 total entries)
Corresponds to 1 classification label of the source (training label),
and 17 unique inputs for the neural network (property value (+ uncertainty on some of them))
'''
features_master_list = [
                        "TTYPE5",             #GLON
                        "TTYPE6",             #GLAT
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
                        "TTYPE41",            #Highest_energy, ***1286 are missing data***
                        "TTYPE21",            #Classification - TRAINING LABEL, NOT AN INPUT
                        "TTYPE30",            #Redshift - TRAINING VALUE, NOT AN INPUT ***1601 are missing data***
                        "TTYPE1"              #Source_Name - UNIQUE OBJECT IDENTIFIER, NOT AN INPUT
                        ]

#CONSIDER ADDING TTYPE32/33 AND TTYPE34/35 - HE_EPeak+unc. and HE_nuFnuPeak+unc. columns

transformations = [
                    "None",
                    "None",
                    "Log+Z",
                    "Log+Z", "Propagate",
                    "Log+Z", "Propagate",
                    "None",
                    "Z-score", "Propagate",
                    "Log+Z",
                    "Z-score", "Propagate",
                    "Z-score", "Propagate", #LP_beta + unc., could go either z-score or log I'm not sure!
                    "None",
                    "None",
                    "Log+Z",
                    "Log+Z",
                    "Log",
                    "Log+Z", "Propagate",
                    "Log+Z",
                    "None",
                    "Log+Z",
                    "None"
                    ]
#test = [i for i, x in enumerate(normalisations) if x != "None"]

def FeatureDisplay(hdu_table, do_hist=False):
    '''
    Displays some basic info about each feature group in the input HDU table 
    
    Inputs: HDU table to be described
            *Optional* Boolean for whether to produce histogram plots for the columns
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
        if do_hist == True:
            fig = plt.figure()
            ax = fig.gca()
            ax.hist(feature_data)
            fig.suptitle(feature_header)
            plt.show()
        
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
        '''
        for x in ["TTYPE32", "TTYPE33", "TTYPE34", "TTYPE35"]:
            print("Feature ", header_array[x], "Num. non-zero left:", len(temp_filtered_table[(temp_filtered_table[header_array[x]] != -np.inf) & (not np.isnan(temp_filtered_table[header_array[x]]))]))            
            print(temp_filtered_table[header_array[x]][10:20], temp_filtered_table[header_array[x]][10:20])
        '''    
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
        
    for i in range(len(filtered_array)):
        class_temp = filtered_array[class_feature][i]
        class_temp = class_temp.lower()
        
        if (class_temp != 'fsrq') and (class_temp != 'bll') and (class_temp != 'bcu'):
            class_temp = 'inval'
            
        filtered_array[class_feature][i] = class_temp
        
    filtered_array = filtered_array[filtered_array[class_feature] != 'inval']
    filtered_array = filtered_array[filtered_array[class_feature] != 'bcu']
    
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

class BayesianNeuralNetwork(PyroModule[nn.Module]):
    '''
    Defines the Bayesian Neural Network that we use for the classification of BCUs and for the redshift predictions
    Declared as a PyroModule mixin class of nn.Module
    
    __init__: Initialises the network with the given network layer sizes and declares the Pyro objects to give the Bayesian behaviour
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
        
        std_scale = global_std_scale
        
        self.L1WMean = PyroParam(torch.zeros_like(self.Layer1.weight))
        self.L1WLogStd = PyroParam(std_scale*torch.ones_like(self.Layer1.weight))
        self.L1WStd = torch.exp(self.L1WLogStd)

        self.L1BMean = PyroParam(torch.zeros_like(self.Layer1.bias))
        self.L1BLogStd = PyroParam(std_scale*torch.ones_like(self.Layer1.bias))
        self.L1BStd = torch.exp(self.L1BLogStd)


        self.L2WMean = PyroParam(torch.zeros_like(self.Layer2.weight))
        self.L2WLogStd = PyroParam(std_scale*torch.ones_like(self.Layer2.weight))
        self.L2WStd = torch.exp(self.L2WLogStd)

        self.L2BMean = PyroParam(torch.zeros_like(self.Layer2.bias))
        self.L2BLogStd = PyroParam(std_scale*torch.ones_like(self.Layer2.bias))
        self.L2BStd = torch.exp(self.L2BLogStd)


        #Declare layer weights and biases as Pyro samples from normal distribution based on previously-declared Pyro parameters
        self.Layer1.weight = PyroSample(dist.Normal(self.L1WMean, self.L1WStd).independent(2))
        self.Layer1.bias = PyroSample(dist.Normal(self.L1BMean, self.L1BStd).independent(1))

        self.Layer2.weight = PyroSample(dist.Normal(self.L2WMean, self.L2WStd).independent(2))
        self.Layer2.bias = PyroSample(dist.Normal(self.L2BMean, self.L2BStd).independent(1))

    def forward(self, input_data):
        #Simply runs through the network stack and returns the outputs
        
        input_data = self.flatten(input_data)
        hidden_data = nn.functional.relu(self.Layer1(input_data))
        output_data = self.Layer2(hidden_data)
        
        return output_data


#-----NEURAL NETWORK INITIALISATION-----#
#Declare whether to sample properties based on their uncertainties, or just include the uncertainties as unique features
sampled_uncertainties = True
print("Using feature values sampled based on uncertainties: ", sampled_uncertainties)

#Define network node sizes, and create a neural network instance for classification plus one for redshift predictions
global_std_scale = -1 #Determines the size of the Gaussian priors for weights and biases

if sampled_uncertainties:
    input_nodes_classification = 17
    input_nodes_redshifts = 18 #Same inputs as classification plus the class probability
else:
    input_nodes_classification = 23
    input_nodes_redshifts = 24
    
hidden_nodes_classification = 32
hidden_nodes_redshifts = 32

output_nodes_classification = 2
output_nodes_redshifts = 1

print(f"Using {hidden_nodes_classification} hidden nodes for classifier, {hidden_nodes_redshifts} hidden nodes for redshift predictor, and a log prior width of {global_std_scale}")

classification_neural_network = BayesianNeuralNetwork(input_nodes_classification, hidden_nodes_classification, output_nodes_classification)
redshifts_neural_network = BayesianNeuralNetwork(input_nodes_redshifts, hidden_nodes_redshifts, output_nodes_redshifts)

def ClassificationModelFunc(input_features, correct_labels=None, sampled_uncertainties=sampled_uncertainties):
    '''
    Runs the model on the given input feature tensor, then sampling a classification from these logits
    Optionally, will call UncertaintySampling to obtain sampled values of uncertain features from a normal distribution with the std. being the feature's associated uncertainty
    Inputs: Tensor of the input features to be passed into the neural network
            *Optional* Tensor of the correct classifications
            Boolean for whether to obtain samples from a normal distribution for features with uncertainties - defined at start of code
            
    Returns: Tensor of the actual correct labels itself (if passed into the function)
             Tensor of the sampled values from the calculated logits if not
    '''
    #print("Start of model:", input_features.shape)
    input_features = torch.squeeze(input_features)
    if sampled_uncertainties:
        input_features_with_samples = UncertaintySampling(input_features)
        
        #print(input_features_with_samples[7, 12:])
        logits = classification_neural_network(input_features_with_samples)
    
    else:
        logits = classification_neural_network(input_features)
                
    pyro.deterministic("probabilities", nn.functional.softmax(logits, dim=-1))
    with pyro.plate("results_class", logits.shape[0]):
        pyro.sample("obs_class", dist.Categorical(logits=logits), obs=correct_labels)

def RedshiftsModelFunc(input_features, correct_redshifts=None, sampled_uncertainties=sampled_uncertainties):
    '''
    Runs the model on the given input feature tensor, then sampling a redshift value from these logits
    Optionally, will call UncertaintySampling to obtain sampled values of uncertain features from a normal distribution with the std. being the feature's associated uncertainty
    Inputs: Tensor of the input features to be passed into the neural network
            *Optional* Tensor of the correct redshift values
            Boolean for whether to obtain samples from a normal distribution for features with uncertainties - defined at start of code
            
    Returns: Tensor of the actual correct labels itself (if passed into the function)
             Tensor of the sampled values from the calculated logits if not
    '''
    #print("Start of model:", input_features.shape)
    input_features = torch.squeeze(input_features)
    if sampled_uncertainties:
        input_features_with_samples = UncertaintySampling(input_features, redshifts=True)
        
        #print(input_features_with_samples[7, 12:])
        predictions = redshifts_neural_network(input_features_with_samples)
    
    else:
        predictions = redshifts_neural_network(input_features)

    estimated_noise_scale = 0.0005 #Noise injected into predictions - very low value treats redshift predictions as a point-prediction
    noise_levels = pyro.sample("sigma", dist.HalfNormal(estimated_noise_scale))
    predictions = predictions.flatten()

    with pyro.plate("results_redshift", predictions.shape[0]):
        pyro.sample("obs_redshift", dist.Normal(predictions, noise_levels), obs=correct_redshifts)
    return predictions

def UncertaintySampling(input_features, redshifts=False):
        if redshifts:
            non_unc_inputs = [0, 1, 2, 7, 10, 15, 16, 17, 18, 19, 22, 23]
        else:
            non_unc_inputs = [0, 1, 2, 7, 10, 15, 16, 17, 18, 19, 22]
            
        #unc_inputs = [3,4, 5,6, 8,9, 11,12, 13,14, 20,21]

        with pyro.plate("uncertainties", input_features.shape[0]):
            Flux1000 = pyro.sample("Flux1000", dist.Normal(input_features[:, 3], input_features[:, 4]))
            Energy_Flux100 = pyro.sample("Energy_Flux100", dist.Normal(input_features[:, 5], input_features[:, 6]))
            PL_Index = pyro.sample("PL_Index", dist.Normal(input_features[:, 8], input_features[:, 9]))
            LP_Index = pyro.sample("LP_Index", dist.Normal(input_features[:, 11], input_features[:, 12]))
            LP_beta = pyro.sample("LP_beta", dist.Normal(input_features[:, 13], input_features[:, 14]))
            Frac_Variability = pyro.sample("Frac_Variability", dist.Normal(input_features[:, 20], input_features[:, 21]))
                
        temp = torch.stack((Flux1000, Energy_Flux100, PL_Index, LP_Index, LP_beta, Frac_Variability))
        temp = torch.transpose(temp, 0, 1)
        
        input_features_with_samples = torch.index_select(input_features, 1, torch.LongTensor(non_unc_inputs))
        input_features_with_samples = torch.cat((input_features_with_samples, temp), dim=1)

        return input_features_with_samples
    
def DataTransformation(input_data, transformations, zscore_means=None, zscore_stds=None):
    '''
    Normalises and scales our features based on the requested transformation list (defined at the program start)
    Master function checks each transformation method, and passes those features to the correct transformation function 
    Z-score means and stds from training dataset can be passed in when separately transforming test dataset (separately transformed to avoid information leakage about test data)
    Inputs: Array containing the input data to be transformed
            Array containing the transformation methods to be used for each feature
            *Optional* Float array containing the z-score means to be applied (if transforming test dataset), or None (if transforming training dataset)
            *Optional* Float array containing the z-score stds to be applied (if transforming test dataset), or None (if transforming training dataset)

    Outputs: Array containing the transformed input data
    '''
    return_array = np.zeros(input_data.shape, dtype=np.float32)
    return_zscore_means = np.zeros(input_data.shape[1], dtype=np.float32)
    return_zscore_stds = np.zeros(input_data.shape[1], dtype=np.float32)
    
    for i in range(input_data.shape[1]):
        temp_array = input_data[:, i]
        method = transformations[i]
        
        if (zscore_means is not None) and (zscore_stds is not None):
            temp_zscore_mean = zscore_means[i]
            temp_zscore_std = zscore_stds[i]
            
        else:
            temp_zscore_mean = None
            temp_zscore_std = None
        
        #for loop will load uncertainties as current feature - want to skip these since we already deal with them!
        if method != "Propagate":
            
            #Prevents overflowing at end of array
            if i < input_data.shape[1]:
                temp_prop_array = None
                next_method = transformations[i+1]
                    
                #Assigns temp_prop_array ONLY if current feature has an associated uncertainty
                if next_method == "Propagate":
                    temp_prop_array = input_data[:, i+1]
                
            #Performs log transform on data and uncertainties
            if method == "Log":
                temp_array, temp_prop_array = DataLogTransform(temp_array, temp_prop_array)
                
            #Performs z-score transform on data and uncertainties
            elif method == "Z-score":
                temp_array, temp_prop_array, temp_zscore_mean, temp_zscore_std = DataZScoring(temp_array, temp_prop_array, temp_zscore_mean, temp_zscore_std)
    
            #Performs both log and z-score transforms on data and uncertainties
            elif method == "Log+Z":
                temp_array, temp_prop_array = DataLogTransform(temp_array, temp_prop_array)
                temp_array, temp_prop_array, temp_zscore_mean, temp_zscore_std = DataZScoring(temp_array, temp_prop_array, temp_zscore_mean, temp_zscore_std)                
    
            elif i==0: #Scales the GLON coordinate
                temp_array = (temp_array-180)/180
                
            elif i==1: #Scales the GLAT coordinate
                temp_array = temp_array/90 
                
            elif method != "None":
                print("Assigned a method that is not accounted for in the transformation function!!!")
                
            #Inserts transformed data into new array
            return_array[:, i] = temp_array
            return_zscore_means[i] = temp_zscore_mean
            return_zscore_stds[i] = temp_zscore_std
    
            #Only inserts propagated uncertainties if feature actually had them to begin with!
            if temp_prop_array is not None:
                return_array[:, i+1] = temp_prop_array
                return_zscore_means[i+1] = temp_zscore_mean
                return_zscore_stds[i+1] = temp_zscore_std

    return return_array, return_zscore_means, return_zscore_stds

def DataLogTransform(input_data, propagation_array):
    '''
    Applies a Log10 transform to the input data
    Propagates this through the uncertainties if they are provided
    Inputs: Array containing the input data to be logged
            Array containing the corresponding uncertainties to be propagated (or None if no uncertainties)
    Returns: Array containing the logged data
             Array containing the Log10 propagated uncertainties (or None if no uncertainties)
    '''    
    temp_array = input_data
    temp_prop_array=None
    
    if propagation_array is not None:
        temp_prop_array = propagation_array
        temp_prop_array = np.abs(temp_prop_array/(np.log(10) * temp_array))
    
    return np.log10(temp_array), temp_prop_array
            
def DataZScoring(input_data, propagation_array, temp_zscore_mean, temp_zscore_std):
    '''
    Applies the Z-score scaling to the input data
    Propagates this through the uncertainties if they are provided
    Test dataset needs to be transformed using the training dataset means/stds to avoid test dataset information leaking into training process
    The z-score means and stds from training are provided along with the test data, or are None values if the input is the training dataset 
    Inputs: Array containing the input data to be z-scored
            Array containing the corresponding uncertainties to be propagated (or None if no uncertainties)
            Float containing the z-score mean to be applied (if transforming test dataset), or None (if transforming training dataset)
            Float containing the z-score std to be applied (if transforming test dataset), or None (if transforming training dataset)

    Returns: Array containing the z-scored data
             Array containing the z-score propagated uncertainties (or None if no uncertainties)
    '''
    temp_array = input_data
    temp_prop_array=None

    if (temp_zscore_mean is not None) and (temp_zscore_std is not None):
        mean = temp_zscore_mean
        std = temp_zscore_std
        
    else:
        mean = np.mean(temp_array)
        std = np.std(temp_array)
    
    if propagation_array is not None:
        temp_prop_array = propagation_array
        temp_prop_array = temp_prop_array/std

    return (temp_array-mean)/std, temp_prop_array, mean, std
            

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
print(np.unique(master_data_array["CLASS"]))
temp_data_array = ClassificationFiltering(master_data_array)
print(np.unique(master_data_array["CLASS"]))

#Split master data into network training data and classification "correct answers"
#Also splits off the source names and the redshifts into separate arrays
train_data_array = np.zeros((len(temp_data_array), len(features_master_list)-3), dtype=np.float32)
train_class_array = np.array(len(temp_data_array), dtype=str)
train_redshift_array_temp = np.array(len(temp_data_array), dtype=np.float32)
train_source_name_array = np.array(len(temp_data_array), dtype=str)

for i in range(len(features_master_list)):
    feature = master_headers_array[features_master_list[i]]
    
    if i==23: #Splits off the classification array
        train_class_array = temp_data_array[feature]
        
        #Convert class strings to numerical values - easier to compare predictions to actual values!
        train_class_array[train_class_array == 'fsrq'] = 1
        train_class_array[train_class_array == 'bll'] = 0
        train_class_array = np.asarray(train_class_array, dtype=int)
        
    elif i==24: #Splits off the redshift array - also have to swap the byte order from 'big-endian' to 'little-endian'
        train_redshift_array_temp = temp_data_array[feature]
        train_redshift_array = train_redshift_array_temp.byteswap().view(train_redshift_array_temp.dtype.newbyteorder('='))
    
    elif i==25: #Splits off the Source_Name array
        train_source_name_array = temp_data_array[feature]
        
    else:
        train_data_array[:, i] = temp_data_array[feature]

#Makes sure that the data, classes, redshift and source names have been split into correctly-sized arrays, then shuffles them (whilst maintaining correspondence)
assert len(train_data_array) == len(train_class_array)
#print(train_data_array[[899, 639, 751, 812, 652], 0], train_class_array[[899, 639, 751, 812, 652]], train_redshift_array[[899, 639, 751, 812, 652]], train_source_name_array[[899, 639, 751, 812, 652]]) #For verifying permutations are working correctly 

shuffled_indices = np.random.permutation(len(train_data_array))
train_data_array = train_data_array[shuffled_indices]
train_class_array = train_class_array[shuffled_indices]
train_redshift_array = train_redshift_array[shuffled_indices]
train_source_name_array = train_source_name_array[shuffled_indices]

#print(shuffled_indices[2:7], train_data_array[2:7, 0], train_class_array[2:7], train_redshift_array[2:7], train_source_name_array[2:7]) #For verifying permutations are working correctly - for rng_seed=1234, should match previous print perfectly

#Split into training set, testing set, and validation set - 80% training set, 20% test set
#No need for validation set since we use k-fold cross-validation
split = int(round(0.8 * len(train_data_array)))
test_data_array, test_class_array, test_redshift_array, test_source_name_array = train_data_array[split:], train_class_array[split:], train_redshift_array[split:], train_source_name_array[split:]
train_data_array, train_class_array, train_redshift_array, train_source_name_array = train_data_array[0:split], train_class_array[0:split], train_redshift_array[0:split], train_source_name_array[0:split]

#Need to use the z-scoring means and stds for the training dataset to transform the test dataset, otherwise model can learn information about the test dataset during training!
untransformed_train_array = train_data_array
untransformed_test_array = test_data_array
train_data_array, zscore_means, zscore_stds = DataTransformation(untransformed_train_array, transformations)
#print(zscore_means, zscore_stds)
test_data_array, zscore_means, zscore_stds = DataTransformation(untransformed_test_array, transformations, zscore_means=zscore_means, zscore_stds=zscore_stds)

def BeforeAndAfterTransformationHistograms():
    for feat in range(5,7):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        plt.subplots_adjust(wspace=0.05)
        ax1.hist(untransformed_train_array[:, feat])
        ax2.hist(train_data_array[:, feat])
        plt.suptitle(f"{hdu_table.header[features_master_list[feat]]} before and after normalisation/scaling")
        #plt.savefig(f"{hdu_table.header[features_master_list[feat]]} before and after normalisation+scaling.png")
        plt.show()
        
#BeforeAndAfterTransformationHistograms()

#Convert data arrays into Torch tensors and initialise DataLoaders 
train_data_tensor, train_class_tensor, train_dataloader = InitialiseDataLoaders(train_data_array, train_class_array, batch_size=32)
#val_data_tensor, val_class_tensor, val_dataloader = InitialiseDataLoaders(val_data_array, val_class_array)
test_data_tensor, test_class_tensor, test_dataloader = InitialiseDataLoaders(test_data_array, test_class_array)

## NEURAL NETWORK TRAINING PROCESS ##
def MCMCMethod(train_data, train_labels, num_samples, ModelFunc):
    '''
    Runs the MCMC method on the provided training data and labels, using the provided model function
    Inputs:
        Tensor containing the training data
        Tensor containing the training labels
        Int for the number of samples to draw for the MCMC run - with warmup included, twice as many samples are produced
        Function for the correct model to be using (i.e: classification or redshift prediction)
    Returns:
        MCMC method object
    '''
    target_accept_prob = 0.8 #Default is 0.8; increasing it decreases step size within NUTS, making sampling more stable (or "robust", according to Pyro documentation)
    print("Using a target acceptance probability of", target_accept_prob)
    nuts_kernel = pyro.infer.NUTS(ModelFunc, jit_compile=False, target_accept_prob=target_accept_prob)
        
    mcmc = pyro.infer.mcmc.MCMC(nuts_kernel, num_samples=num_samples)#, warmup_steps=min(num_samples, 100))
    mcmc.run(train_data, train_labels)
        
    return mcmc

def ClassMCMCAccuracy(test_data, test_classes, samples_no_unc, return_metrics, run_number=10):
    '''
    Determines the accuracy of the model trained using MCMC by comparing model predictions on the test data to the true labels
    Inputs:
        MCMC method object - irrelevant if samples are passed into function
        Tensor containing the test data
        Tensor containing the test labels
        *Optional* Dictionary containing the MCMC samples if already obtained, or None if samples not passed into ClassMCMCPlotting function
        *Optional* Int specifying the number of times to apply the samples to the provided test data
    Returns:
        Array containing the predicted labels for each object by each sample
        Array containing the raw class probabilities for each object by each sample
    '''
    test_classes = test_classes.cpu().numpy()
    '''
    #Only want posterior samples for the network weights and biases - input variables should be resampled!
    if samples_no_unc == None:
        samples = mcmc.get_samples()
        samples_no_unc = {}
        network_keys = ['Layer1.weight', 'Layer1.bias', 'Layer2.weight', 'Layer2.bias']
        for x in network_keys:
            samples_no_unc[x] = samples[x]
     '''   
    accuracy_rates = np.zeros(run_number)
    f1_scores = np.zeros(run_number)
    brier_scores = np.zeros(run_number)
    auc_scores=np.zeros(run_number)

    for i in range(run_number):        
        predictive = pyro.infer.Predictive(model=ClassificationModelFunc, posterior_samples=samples_no_unc, return_sites={"obs_class", "probabilities"})
        output = predictive(test_data)
        preds, probs = output['obs_class'].cpu().numpy(), output['probabilities'].squeeze().cpu().numpy()
        
        correct_number=0
        f1_score_total=0
        brier_score_total=0
        auc_total=0
        '''
        for x in range(preds.shape[0]):
            correct_number += (preds[x] == test_classes)
            f1_score_total += sklearn.metrics.f1_score(test_classes, preds[x])
            brier_score_total += np.sum((probs[x, :, 1]-test_classes)**2)
            auc_total += sklearn.metrics.roc_auc_score(test_classes, np.mean(probs, axis=0)[:, 1])
        '''
        avg_probs = np.mean(probs, axis=0)
        avg_preds = np.argmax(avg_probs, axis=1)
        
        correct_number += np.sum((avg_preds == test_classes))
        f1_score_total += sklearn.metrics.f1_score(test_classes, avg_preds)
        brier_score_total += np.sum((avg_probs[:, 1]-test_classes)**2)
        auc_total += sklearn.metrics.roc_auc_score(test_classes, avg_probs[:, 1])
        
        num_objects = preds.shape[1]
        
        accuracy_rate = 100*(correct_number/num_objects)
        #print("Accuracy rate:", np.mean(accuracy_rate), "%")
        #print("Average F1 score:", f1_score_total)
        #print("Average Brier score:", brier_score_total/(preds.shape[1]))
        #print("Average AUC score:", auc_total)
        
        
        accuracy_rates[i] = np.mean(accuracy_rate)
        f1_scores[i] = f1_score_total
        brier_scores[i] = brier_score_total/(preds.shape[1])
        auc_scores[i] = auc_total
        
    #print(avg_preds, test_classes, np.sum(avg_preds==test_classes))
    
    print("Mean accuracy rate for sample set:", np.mean(accuracy_rates), "%")
    print("Mean F1 score for sample set:", np.mean(f1_scores))
    print("Mean Brier score for sample set:", np.mean(brier_scores))
    print("Mean AUC score for sample set:", np.mean(auc_total))

    if return_metrics:
        return preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores
    else:
        return preds, probs

def ClassMCMCPlotting(mcmc, test_data, test_classes, samples=None, plots=True, return_metrics=False):
    '''
    Produces some plots using the MCMC data (e.g: predicted labels histogram, trace plots of Layer 2 weights, entropy of predictions)
    Inputs:
        MCMC method object
        Tensor containing the test data
        Tensor containing the test labels
        *Optional* Dictionary containing the MCMC samples if already obtained - just used to pass through to the ClassMCMCAccuracy function (defaults to None if samples not provided)
    '''    
    if return_metrics:
        preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores = ClassMCMCAccuracy(test_data, test_classes, samples, return_metrics)
    else:
        preds, probs = ClassMCMCAccuracy(test_data, test_classes, samples, return_metrics)
        
    fsrqs = np.where(test_classes.cpu().numpy() == 1)
    blls = np.where(test_classes.cpu().numpy() == 0)
    
    mean_probs = np.mean(probs, axis=0)
    mean_pred = np.argmax(mean_probs, axis=1)
                
    if plots == True:
        plt.hist(np.squeeze(mean_probs[fsrqs, 1]), bins=10, label="FSRQs", color='red', alpha=0.7)
        plt.hist(1-np.squeeze(mean_probs[blls, 0]), bins=10, label="BL Lacs", color='blue', alpha=0.7)
        plt.plot([np.percentile(np.squeeze(mean_probs[fsrqs, 1]), 20), np.percentile(np.squeeze(mean_probs[fsrqs, 1]), 20)], [0,50], label="FSRQ 80th Percentile", ls = '--', color="Black")
        plt.plot([np.percentile(1-np.squeeze(mean_probs[blls, 0]), 80), np.percentile(1-np.squeeze(mean_probs[blls, 0]), 80)], [0,50], label="BL Lac 80th Percentile", ls = '--', color="Magenta")
        plt.xlabel("Mean probability")
        plt.ylabel("Box density")
        plt.title("Mean prediction per object type (BL Lac = 0, FSRQ = 1)")
        plt.legend()
        plt.show()
        
        entropy = -((mean_probs[:, 0]*np.log(mean_probs[:, 0]+1e-9)) + ((mean_probs[:, 1])*np.log(mean_probs[:, 1]+1e-9)))
        
        plt.hist(entropy, bins=50)
        plt.title("Entropy of each object prediction")
        plt.show()
        
        if mcmc is not None:
            trace_params = mcmc.get_samples()['Layer2.weight'].cpu().numpy()
            trace_params = np.array([trace_params])
            az.plot_trace(trace_params)
            plt.title("Posterior distributions for Layer 2 weights")
            plt.show()
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_classes, mean_probs[:, 1])
        #print("THRESHOLDS: ", thresholds)
        auc_score = sklearn.metrics.roc_auc_score(test_classes, mean_probs[:, 1])
        plt.plot(fpr, tpr, label='ROC Curve for model')
        plt.plot([0, 1], [0, 1], ls='--', color='gray', alpha=0.6, label='Random guesses curve')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"ROC curve for model; AUC score = {auc_score:.3f}")
        plt.legend()
        plt.show()
        
        plt.scatter(np.mean(probs[:, :, 1], axis=0), np.std(probs[:, :, 1], axis=0), color='red', marker='x', alpha=0.8)
        plt.title("Uncertainty (standard deviation) in FSRQ probability against mean FSRQ probability per object")
        plt.xlabel("Mean FSRQ probability")
        plt.ylabel("Uncertainty in FSRQ probability")
        plt.show()
        '''
        plt.scatter(np.mean(probs[:, :, 0], axis=0), np.std(probs[:, :, 0], axis=0), color='blue', marker='x')
        plt.title("Uncertainty (standard deviation) in BLL probability against mean BLL probability per object")
        plt.xlabel("Mean BLL probability")
        plt.ylabel("Uncertainty in BLL probability")
        plt.show()
        '''
        #plt.scatter(mean_pred, test_data[:, 14]) #PL_Index against uncertainty
        
    if return_metrics:
        return accuracy_rates, f1_scores, brier_scores, auc_scores
'''
#Temporary code for calculating metrics for separate folds
if cross_validation_k_class > 1:
    #CV is useful to show that the model generalises well to unseen data; (hopefully) provides evidence that the precise split in the dataset is unimportant
    #skf = sklearn.model_selection.StratifiedKFold(cross_validation_k_class) #Stratified K-fold CV keeps class compositions equal between fold splits
    
    master_accuracies = np.zeros(cross_validation_k_class)
    master_f1s = np.zeros(cross_validation_k_class)
    master_briers = np.zeros(cross_validation_k_class)
    master_aucs = np.zeros(cross_validation_k_class)
    for fold, (train_data, test_data) in enumerate(skf.split(train_data_tensor, train_class_tensor)):
        print(f"Fold {fold+1}")
        temp_val_data = train_data_tensor[test_data]
        temp_val_classes = train_class_tensor[test_data]
        
        temp_train_data = train_data_tensor[train_data]
        temp_train_classes = train_class_tensor[train_data]
            
        #mcmc = MCMCMethod(temp_train_data, temp_train_classes, num_samples, ClassificationModelFunc)
        #list_mcmcs_class.append(mcmc)
        accuracies, f1_scores, brier_scores, auc_scores = ClassMCMCPlotting(list_mcmcs_class[fold], temp_val_data, temp_val_classes, return_metrics=True, plots=False)
        master_accuracies[fold] = np.mean(accuracies)
        master_f1s[fold] = np.mean(f1_scores)
        master_briers[fold] = np.mean(brier_scores)
        master_aucs[fold] = np.mean(auc_scores)
    
    print("Average accuracy from each fold: ", master_accuracies)
    print("Average accuracy over all folds: ", np.mean(master_accuracies))
    print("Average F1-score from each fold: ", master_f1s)
    print("Average F1-score over all folds: ", np.mean(master_f1s))
    print("Average Brier score from each fold: ", master_briers)
    print("Average Brier score over all folds: ", np.mean(master_briers))
    print("Average AUC from each fold: ", master_aucs)
    print("Average AUC over all folds: ", np.mean(master_aucs))
'''
    
def DictSplit(samples_dict, cross_validation_k_number=5):
    '''
    Splits a full sample dictionary into k separate dictionaries
    Inputs: 
        Dictionary containing all samples (and keys)
    Returns: 
        List of dictionaries containing the samples for the network weights and biases
    '''
    split_sample_dictionaries = []
    for j in range(cross_validation_k_number):
        split_sample_dictionaries.append({})
        
    sample_number = samples_dict['Layer2.bias'].shape[0]
    network_keys = ['Layer1.weight', 'Layer1.bias', 'Layer2.weight', 'Layer2.bias']
    
    for x in network_keys:
        temp = samples_dict[x]
        splits = np.split(temp, cross_validation_k_number, axis=0)
        #print(f"{x}", splits)
        for i in range(cross_validation_k_number):
            split_sample_dictionaries[i][x] = splits[i]
            
    return split_sample_dictionaries

def SaveSamples(sample_dictionary, file_name='temp_samples_dict.npy'):
    print(f"Saving samples to file {file_name}")
    np.save(f'{file_name}', sample_dictionary)
    
def LoadSamples(file_name):
    print(f"Loading samples from file {file_name}")
    return np.load(f'{file_name}', allow_pickle=True).item()

def RetrieveBCUs(input_master_array):
    class_feature = "CLASS"
    filtered_array = input_master_array
        
    for i in range(len(filtered_array)):
        class_temp = filtered_array[class_feature][i]
        class_temp = class_temp.lower()
        
        if (class_temp != 'bcu'):
            class_temp = 'inval'
            
        filtered_array[class_feature][i] = class_temp
    
    filtered_array = filtered_array[filtered_array[class_feature] == 'bcu']
    return filtered_array

def ClassifyingBCUs(input_master_array, sample_set=None, zscore_means=zscore_means, zscore_stds=zscore_stds, plots=True):
    print(np.unique(input_master_array["CLASS"]))
    master_bcu_array = RetrieveBCUs(input_master_array)
    #print(np.unique(master_bcu_array["CLASS"]))
    
    bcu_data_array = np.zeros((len(master_bcu_array), len(features_master_list)-3), dtype=np.float32)
    bcu_class_array = np.array(len(master_bcu_array), dtype=str)
    bcu_redshift_array_temp = np.array(len(master_bcu_array), dtype=np.float32)
    bcu_source_name_array = np.array(len(master_bcu_array), dtype=str)
    
    for i in range(len(features_master_list)):
        feature = master_headers_array[features_master_list[i]]
        
        if i==23: #Splits off the classification array
            bcu_class_array = master_bcu_array[feature]
            
            #Convert class strings to numerical values - easier to compare predictions to actual values!
            bcu_class_array[bcu_class_array == 'bcu'] = 2
            bcu_class_array = np.asarray(bcu_class_array, dtype=int)
            
        elif i==24: #Splits off the redshift array - also have to swap the byte order from 'big-endian' to 'little-endian'
            bcu_redshift_array_temp = master_bcu_array[feature]
            bcu_redshift_array = bcu_redshift_array_temp.byteswap().view(bcu_redshift_array_temp.dtype.newbyteorder('='))

            
        elif i==25: #Splits off the Source_Name array
            bcu_source_name_array = master_bcu_array[feature]
            
        else:
            bcu_data_array[:, i] = master_bcu_array[feature]
            
    bcu_data_array, zscore_means, zscore_stds = DataTransformation(bcu_data_array, transformations, zscore_means=zscore_means, zscore_stds=zscore_stds)
    bcu_data_tensor = torch.tensor(bcu_data_array)
    
    #print(bcu_data_tensor.shape)
    
    if sample_set == None:
        samples_file_name = 'temp_samples_dict.npy'
        sample_set = LoadSamples(samples_file_name)
    
    predictive = pyro.infer.Predictive(model=ClassificationModelFunc, posterior_samples=sample_set, return_sites={"obs_class", "probabilities"})
    output = predictive(bcu_data_tensor)
    preds, probs = output['obs_class'].cpu().numpy(), output['probabilities'].squeeze().cpu().numpy()
    
    avg_probs = np.mean(probs, axis=0)
    mean_pred = np.argmax(avg_probs, axis=1)
    
    if plots==True:
        plt.hist(np.squeeze(avg_probs[:, 1]), label='FSRQ Probability bins', bins=20, color='red')
        plt.plot([0.5, 0.5], [0, 75], color='black', label='50% probability threshold')
        plt.title("Histogram of the FSRQ probabilities of classified BCUs")
        plt.ylabel("Number of objects")
        plt.xlabel("FSRQ Probability")
        plt.legend(loc='upper right')
        plt.show()
        
        plt.scatter(np.mean(probs[:, :, 1], axis=0), np.std(probs[:, :, 1], axis=0), color='red', marker='x', alpha=0.8)
        plt.title("Uncertainty (standard deviation) in FSRQ probability against mean FSRQ probability per BCU object")
        plt.xlabel("Mean FSRQ probability")
        plt.ylabel("Uncertainty in FSRQ probability")
        plt.show()
        
    #print(np.unique(mean_pred, return_counts=True))
    return bcu_data_tensor, probs, bcu_redshift_array, bcu_source_name_array

def ListMergingAndTrimmingMCMCs(mcmc_list):
    #Trim out unnecessary columns and combine the MCMC samples from each fold into one large dictionary    
    all_samples = {}
    network_keys = ['Layer1.weight', 'Layer1.bias', 'Layer2.weight', 'Layer2.bias']
    for x in network_keys:
        for j in range(len(mcmc_list)):
            mcmc = mcmc_list[j]
            if j == 0:
                temp = mcmc.get_samples()[x]
            else:
                temp = torch.cat((temp, mcmc.get_samples()[x]), dim=0)
            
        all_samples[x] = temp
        
    return all_samples
    

#losses_array = SVIMethod()

###CAN RUN THESE BEFORE ANY MCMC STUFF TO INVESTIGATE INITIAL PREDICTIVE BEHAVIOUR OF CLASSIFICATION NETWORK###
###ADD A RETURN TO THE PYRO.DETERMINISTIC OF THE PROBABILITIES BEFORE RUNNING THESE, SO PROBS ARE RETURNED###
#test = ClassificationModelFunc(train_data_tensor)
#plt.hist(test.detach().numpy())

list_mcmcs_class = []
num_samples = 500 #Per cross_validation run, including warmup we have 2*cross_validation_k*num_samples done in total
cross_validation_k_class = 1 #Number of folds to make, number of cross-validation runs to perform. Setting it to 1 just trains one model on the whole training set, and applies it to the test set

if cross_validation_k_class > 1:
    #CV is useful to show that the model generalises well to unseen data; (hopefully) provides evidence that the precise split in the dataset is unimportant
    skf = sklearn.model_selection.StratifiedKFold(cross_validation_k_class) #Stratified K-fold CV keeps class compositions equal between fold splits
    
    for fold, (train_data, test_data) in enumerate(skf.split(train_data_tensor, train_class_tensor)):
        print(f"Fold {fold+1}")
        temp_val_data = train_data_tensor[test_data]
        temp_val_classes = train_class_tensor[test_data]
        
        temp_train_data = train_data_tensor[train_data]
        temp_train_classes = train_class_tensor[train_data]
            
        mcmc = MCMCMethod(temp_train_data, temp_train_classes, num_samples, ClassificationModelFunc)
        list_mcmcs_class.append(mcmc)
        ClassMCMCPlotting(list_mcmcs_class[fold], temp_val_data, temp_val_classes, return_metrics=False)

else:
    #Training a single model on all data at once gives the 'most optimal' sample set
    mcmc = MCMCMethod(train_data_tensor, train_class_tensor, num_samples, ClassificationModelFunc)
    list_mcmcs_class.append(mcmc)


all_samples_class = ListMergingAndTrimmingMCMCs(list_mcmcs_class)
#all_samples_class = LoadSamples('temp_samples_class_dict.npy')

def RedshiftPlotting(input_data_tensor, input_redshift_tensor, samples, plots=False):
    predictive = pyro.infer.Predictive(model=RedshiftsModelFunc, posterior_samples=samples, return_sites={"obs_redshift", "sigma"})
    output = predictive(input_data_tensor)
    redshift_preds = output['obs_redshift'].cpu().numpy().squeeze()
    #plt.hist(output['sigma'])
    
    avg_redshift_preds = np.mean(redshift_preds, axis=0)
    std_redshift_preds = np.std(redshift_preds, axis=0)
    
    redshifts_good_fit_indices = np.where(std_redshift_preds < 2.5)
    avg_redshift_preds = avg_redshift_preds[redshifts_good_fit_indices]
    std_redshift_preds = std_redshift_preds[redshifts_good_fit_indices]
    input_redshift_tensor = input_redshift_tensor[redshifts_good_fit_indices]
    
    input_redshift_array = input_redshift_tensor.cpu().numpy()
    RMSE = np.sqrt(np.sum(((avg_redshift_preds - input_redshift_array)**2)/input_redshift_array)/len(input_redshift_array))
    print("RMSE:", RMSE)
        
    if plots:
        index_to_plot = 6
        plt.hist(redshift_preds[:, index_to_plot], bins=20)
        plt.title(f"Histogram of predictions over all samples for object, index {index_to_plot}, in dataset")
        plt.show()
        plt.hist(avg_redshift_preds, density=True, alpha=0.6, label="Predicted", bins=20)
        plt.hist(input_redshift_tensor, density=True, alpha=0.6, label="Actual", bins=20)
        plt.legend()
        plt.title(f"Histogram of the predicted values and actual values of redshift for the dataset of {input_redshift_tensor.shape[0]} sources")
        plt.show()
        plt.errorbar(input_redshift_tensor, avg_redshift_preds, yerr=std_redshift_preds, fmt='x')#, marker='x')
        plt.plot([0, 3], [0, 3], ls='--', color='black', label='Perfect prediction accuracy')
        plt.xlabel("Actual redshift values")
        plt.ylabel("Predicted redshift values")
        plt.title(f"Scatter plot of predicted values against actual values of redshift for the dataset of {input_redshift_tensor.shape[0]} sources")
        plt.legend()
        plt.show()

def DataFormattingForRedshifts(input_data_tensor, input_class_tensor, input_redshift_array, class_samples):
    preds, probs = ClassMCMCAccuracy(input_data_tensor, input_class_tensor, class_samples, return_metrics=False, run_number=1)
    mean_probs_tensor = torch.tensor(np.mean(probs[:, :, 1], axis=0)) #Taking FSRQ probabilities as inputs
    temp_data_tensor = torch.cat((input_data_tensor, mean_probs_tensor.unsqueeze(1)), dim=1)

    known_redshift_indices = np.where(input_redshift_array != -np.inf)
    data_tensor_for_redshifts = temp_data_tensor[known_redshift_indices, :]
    known_redshifts_tensor = torch.tensor(input_redshift_array[known_redshift_indices])
    
    return data_tensor_for_redshifts, known_redshifts_tensor
    
'''
#Training data reduction
overflow_preds, train_class_mean_probs = ClassMCMCAccuracy(train_data_tensor, train_class_tensor, all_samples_class, return_metrics=False, run_number=1)
train_class_mean_probs_tensor = torch.tensor(np.mean(train_class_mean_probs[:, :, 1], axis=0))
redshift_train_data_tensor = torch.cat((train_data_tensor, train_class_mean_probs_tensor.unsqueeze(1)), dim=1)

known_redshift_indices = np.where(train_redshift_array != -np.inf)
redshift_train_data_tensor = redshift_train_data_tensor[known_redshift_indices, :]
train_redshift_tensor = torch.tensor(train_redshift_array[known_redshift_indices])


#Testing data reduction
overflow_preds_test, test_class_mean_probs = ClassMCMCAccuracy(test_data_tensor, test_class_tensor, all_samples_class, return_metrics=False, run_number=1)
test_class_mean_probs_tensor = torch.tensor(np.mean(test_class_mean_probs[:, :, 1], axis=0))
redshift_test_data_tensor = torch.cat((test_data_tensor, test_class_mean_probs_tensor.unsqueeze(1)), dim=1)

known_redshift_indices_test = np.where(test_redshift_array != -np.inf)
redshift_test_data_tensor = redshift_test_data_tensor[known_redshift_indices_test, :]
test_redshift_tensor = torch.tensor(test_redshift_array[known_redshift_indices_test])


#BCU data reduction
bcu_temp_data_tensor, bcu_class_mean_probs, bcu_redshift_array, bcu_source_name_array = ClassifyingBCUs(master_data_array, sample_set=all_samples_class, plots=False)
bcu_class_mean_probs_tensor = torch.tensor(np.mean(bcu_class_mean_probs[:, :, 1], axis=0))
bcu_data_tensor = torch.cat((bcu_temp_data_tensor, bcu_class_mean_probs_tensor.unsqueeze(1)), dim=1)

known_bcu_redshift_indices = np.where(bcu_redshift_array != -np.inf)
known_bcu_data_tensor, known_bcu_class_mean_probs, known_bcu_redshifts, known_bcu_source_name_array = bcu_data_tensor[known_bcu_redshift_indices, :], bcu_class_mean_probs_tensor[known_bcu_redshift_indices], bcu_redshift_array[known_bcu_redshift_indices], bcu_source_name_array[known_bcu_redshift_indices]
known_bcu_redshifts_tensor = torch.tensor(known_bcu_redshifts)
#known_bcu_data_tensor = torch.squeeze(known_bcu_data_tensor)
#print(known_bcu_data_tensor.shape)
#print(known_bcu_class_mean_probs)
'''

train_data_tensor_redshifts, train_redshifts_tensor = DataFormattingForRedshifts(train_data_tensor, train_class_tensor, train_redshift_array, all_samples_class)
test_data_tensor_redshifts, test_redshifts_tensor = DataFormattingForRedshifts(test_data_tensor, test_class_tensor, test_redshift_array, all_samples_class)

#BCU data reduction
bcu_temp_data_tensor, bcu_class_mean_probs, bcu_redshift_array, bcu_source_name_array = ClassifyingBCUs(master_data_array, sample_set=all_samples_class, plots=False)
bcu_class_mean_probs_tensor = torch.tensor(np.mean(bcu_class_mean_probs[:, :, 1], axis=0))
bcu_data_tensor = torch.cat((bcu_temp_data_tensor, bcu_class_mean_probs_tensor.unsqueeze(1)), dim=1)

known_bcu_redshift_indices = np.where(bcu_redshift_array != -np.inf)
known_bcu_data_tensor, known_bcu_class_mean_probs, known_bcu_redshifts, known_bcu_source_name_array = bcu_data_tensor[known_bcu_redshift_indices, :], bcu_class_mean_probs_tensor[known_bcu_redshift_indices], bcu_redshift_array[known_bcu_redshift_indices], bcu_source_name_array[known_bcu_redshift_indices]
known_bcu_redshifts_tensor = torch.tensor(known_bcu_redshifts)


list_mcmcs_redshift = []
#Only attempts to run the redshift predictions when the full dataset has been used in training for classifications        
if cross_validation_k_class == 1:
    mcmc = MCMCMethod(train_data_tensor_redshifts, train_redshifts_tensor, num_samples, RedshiftsModelFunc)
    list_mcmcs_redshift.append(mcmc)
    
    all_samples_redshift = ListMergingAndTrimmingMCMCs(list_mcmcs_redshift)
    #all_samples_redshift = LoadSamples('temp_samples_redshift_dict.npy')
    
    #Training data predictions
    RedshiftPlotting(train_data_tensor_redshifts, train_redshifts_tensor, all_samples_redshift, plots=True)
    
    #Test data predictions
    RedshiftPlotting(test_data_tensor_redshifts, test_redshifts_tensor, all_samples_redshift, plots=True)
    
    #BCU Predictions
    RedshiftPlotting(known_bcu_data_tensor, known_bcu_redshifts_tensor, all_samples_redshift, plots=True)

    
finish = time.time()
print("Run time:", finish-start, "seconds")
print("Used feature values sampled based on uncertainties: ", sampled_uncertainties)

#SaveSamples(all_samples_class, file_name='temp_samples_class_dict.npy')
#SaveSamples(all_samples_redshift, file_name='temp_samples_redshift_dict.npy')

#loaded_samples = LoadSamples('temp_samples_dict.npy')

#time.sleep(10)
#os.system("shutdown.exe /h")

'''
#Bespoke code to run through previously-generated sample files during node testing and calculate diagnostic metrics
#Need to redefine the model each time to change the model's node number so that pyro.predictive runs correctly
nodes_to_test = [2, 4, 8, 16, 32, 64, 128, 256]

master_accuracies = np.zeros(len(nodes_to_test))
master_f1 = np.zeros(len(nodes_to_test))
master_brier = np.zeros(len(nodes_to_test))
master_auc = np.zeros(len(nodes_to_test))

for x in range(len(nodes_to_test)):
    temp_node_num = nodes_to_test[x]
    print(f"\n {temp_node_num} hidden nodes")
    
    temp_neural_network = BayesianNeuralNetwork(input_nodes_classification, temp_node_num, output_nodes_classification)
    
    def ClassificationModelFunc(input_features, correct_labels=None, sampled_uncertainties=sampled_uncertainties):
        
        #Runs the model on the given input feature tensor, then sampling a classification from these logits
        #Optionally will use sampled values of uncertain features from a normal distribution with the std. being the feature's associated uncertainty
        #
        #Inputs: Tensor of the input features to be passed into the neural network
        #        *Optional* Tensor of the correct classifications
        #        Boolean for whether to obtain samples from a normal distribution for features with uncertainties
                
        #Returns: Tensor of the actual correct labels itself (if passed into the function)
        #         Tensor of the sampled values from the calculated logits if not
        
        #print("Start of model:", input_features.shape)
        input_features = torch.squeeze(input_features)
        if sampled_uncertainties:
            non_unc_inputs = [0, 1, 2, 7, 10, 15, 16, 17, 18, 19, 22]
            #unc_inputs = [3,4, 5,6, 8,9, 11,12, 13,14, 20,21]
            
            with pyro.plate("uncertainties", input_features.shape[0]):
                Flux1000 = pyro.sample("Flux1000", dist.Normal(input_features[:, 3], input_features[:, 4]))
                Energy_Flux100 = pyro.sample("Energy_Flux100", dist.Normal(input_features[:, 5], input_features[:, 6]))
                PL_Index = pyro.sample("PL_Index", dist.Normal(input_features[:, 8], input_features[:, 9]))
                LP_Index = pyro.sample("LP_Index", dist.Normal(input_features[:, 11], input_features[:, 12]))
                LP_beta = pyro.sample("LP_beta", dist.Normal(input_features[:, 13], input_features[:, 14]))
                Frac_Variability = pyro.sample("Frac_Variability", dist.Normal(input_features[:, 20], input_features[:, 21]))
            
            temp = torch.stack((Flux1000, Energy_Flux100, PL_Index, LP_Index, LP_beta, Frac_Variability))
            temp = torch.transpose(temp, 0, 1)
            
            input_features_with_samples = torch.index_select(input_features, 1, torch.LongTensor(non_unc_inputs))
            input_features_with_samples = torch.cat((input_features_with_samples, temp), dim=1)
            
            #print(input_features_with_samples[7, 12:])
            logits = temp_neural_network(input_features_with_samples)
        
        else:
            logits = temp_neural_network(input_features)
                        
        pyro.deterministic("probabilities", nn.functional.softmax(logits, dim=-1))
        
        with pyro.plate("results", logits.shape[0]):
            pyro.sample("obs_class", dist.Categorical(logits=logits), obs=correct_labels)
            
    
    temp_samples = LoadSamples(f"individual_{temp_node_num}_nodes.npy")
    preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores = ClassMCMCAccuracy(test_data_tensor, test_class_tensor, temp_samples, return_metrics=True, run_number=5)
        
    master_accuracies[x] = np.mean(accuracy_rates)
    master_f1[x] = np.mean(f1_scores)
    master_brier[x] = np.mean(brier_scores)
    master_auc[x] = np.mean(auc_scores)
    
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)

ax1.scatter(np.log2(nodes_to_test), master_accuracies, color='red', marker='x')
fig.suptitle("Average accuracy rate, F1-score, and Brier score against $log_2$ of the hidden node number")
ax1.set_ylabel("Accuracy / %")

ax2.scatter(np.log2(nodes_to_test), master_f1, color='green', marker='x')

ax2.set_ylabel("F1-Score")

ax3.scatter(np.log2(nodes_to_test), master_brier, color='blue', marker='x')
ax3.set_ylabel("Brier Score")

ax4.scatter(np.log2(nodes_to_test), master_auc, color='purple', marker='x')
ax4.set_ylabel("AUROC")

ax4.set_xlabel("$Log_2$ of the number of hidden nodes")
plt.show()
'''

'''
#Takes the full samples dataset, splits the samples into each individual fold, plots AUC curves for each fold
auc_total=0
file_to_load = 'temp_samples_dict.npy'
imported_samples = LoadSamples(file_to_load)
split_samples = DictSplit(imported_samples)

for fold, (train_data, test_data) in enumerate(skf.split(train_data_tensor, train_class_tensor)):    
    temp_val_data = train_data_tensor[test_data]
    temp_val_classes = train_class_tensor[test_data]
    
    preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores = ClassMCMCAccuracy(temp_val_data, temp_val_classes, split_samples[fold], return_metrics=True, run_number=1)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(temp_val_classes, np.mean(probs, axis=0)[:, 1])
    auc_total += sklearn.metrics.roc_auc_score(temp_val_classes, np.mean(probs, axis=0)[:, 1])
    
    if fold != cross_validation_k_class - 1:
        plt.plot(fpr, tpr, color='purple', alpha=0.3)
        
plt.plot(fpr, tpr, label='ROC Curve for model', color='purple', alpha=0.3)
plt.plot([0, 1], [0, 1], ls='--', color='gray', alpha=0.7, label='Random guesses curve')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title(f"ROC curve for model; average AUC score = {auc_total/cross_validation_k_class:.3f}")
plt.legend()    
plt.show()
'''