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
import scipy
import multiprocessing
import sys

#Functionally, makes sure that Pytorch and Pyro have been imported and linked correctly
assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

start=time.time()

#Set RNG seed for reproducibility and clear paramater store to make sure we start fresh!
rng_seed = 1234
pyro.set_rng_seed(rng_seed)
pyro.clear_param_store()

if __name__=="__main__":
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
                    "Log+Z",
                    "Z-score", "Propagate",
                    "Log+Z",
                    "None",
                    "None",
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
        print("\nFeature:", feature_header, "\nNum. elements:", len(feature_data), "\nNum. non-zero:", np.count_nonzero(feature_data))
        try: print(np.min(feature_data), np.max(feature_data))
        except: print("Not correct data type for min. and max. values")
        print(np.unique(feature_data, return_counts=True))
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
        #print(np.unique(temp_filtered_table["CLASS"].lower(), return_counts=True))
        '''
        for x in ["TTYPE32", "TTYPE33", "TTYPE34", "TTYPE35"]:
            print("Feature ", header_array[x], "Num. non-zero left:", len(temp_filtered_table[(temp_filtered_table[header_array[x]] != -np.inf) & (not np.isnan(temp_filtered_table[header_array[x]]))]))            
            print(temp_filtered_table[header_array[x]][10:20], temp_filtered_table[header_array[x]][10:20])
        '''    
        data_array = temp_filtered_table

    return header_array, data_array

def MissingDataImputation(hdu_table, impute_mean_values=False, impute_median_values=True):
    '''
    Imputes missing data values with the means or medians from their respective datasets
    Also filters out sources with any associated flag *LIKELY ONLY TEMPORARY*
    
    Inputs: HDU table
            *Optional* Boolean depending on whether to filter bad sources or not (default True)
    Returns: HDU headers
             HDU data as separate arrays (NOT the full combined HDU table)
             
    '''
    problematic_features = ["TTYPE36", "TTYPE37", "TTYPE39", "TTYPE40", "TTYPE41"]
    
    header_array = hdu_table.header
    data_array = hdu_table.data
    
    temp_imputed_table = data_array
    for x in problematic_features:
        feature_header = header_array[x]
        existing_features = temp_imputed_table[(temp_imputed_table[feature_header] != 0) & (temp_imputed_table[feature_header] != -np.inf)][feature_header]
        
        #Impute with mean values
        if impute_mean_values == True:
            existing_features_imputation_value = np.mean(existing_features)

        #Impute with median values
        elif impute_median_values == True:
            existing_features_imputation_value = np.median(existing_features)
       
        for i in range(temp_imputed_table[feature_header].shape[0]):
            if (temp_imputed_table[feature_header][i] == 0) or (temp_imputed_table[feature_header][i] == -np.inf):
                temp_imputed_table[feature_header][i] = existing_features_imputation_value
                                        
        #Since we are imputing synchrotron information, we need to update the SED class column too with the correct class type
        if x == "TTYPE36":
            sed_class_header = header_array["TTYPE31"]
            if existing_features_imputation_value < 1e14:
                imputation_sed_class = 'LSP'
                
            elif existing_features_imputation_value < 1e15:
                imputation_sed_class = 'ISP'
                
            else:
                imputation_sed_class = 'HSP'
            
            #print(np.unique(temp_imputed_table[sed_class_header], return_counts=True))

            for i in range(temp_imputed_table[sed_class_header].shape[0]):
                if temp_imputed_table[sed_class_header][i] == '':
                    temp_imputed_table[sed_class_header][i] = imputation_sed_class
                
            #print(np.unique(temp_imputed_table[sed_class_header], return_counts=True))
    
    flag_feature = header_array["TTYPE20"]
    temp_imputed_table = temp_imputed_table[(temp_imputed_table[flag_feature] == np.int16(0))]
    
    data_array = temp_imputed_table
    
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
    Inputs: 2D array OR 2D tensor for the network's input features
            1D array OR 1D tensor for the actual source classifications
            *Optional* Integer for batch size to use for DataLoader
            *Optional* Boolean for whether the DataLoader should shuffle the input orders
    Outputs: Tensor for the input features
             Tensor for the actual classifications
             DataLoader object containing the input features and classifications
    '''
    if type(feature_array) == torch.Tensor:
        feature_tensor = feature_array
    else:
        feature_tensor = torch.from_numpy(feature_array)
        
    if type(class_array) == torch.Tensor:
        class_tensor = class_array
    else:
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
    def __init__(self, input_dim, hidden_dim, output_dim, prior_scale):
        #Runs the initialisation of the network's PyTorch superclass (nn.Module)
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        #Initialises the network layers as PyroModules
        self.Layer1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.Layer2 = PyroModule[nn.Linear](hidden_dim, output_dim)
        
        #Declare parameters for each network weight as Pyro parameters, mean and std. of a normal distribution
        #########Log stds. declared since it helps with training#########
        
        std_scale = prior_scale
        
        self.L1WMean = 0*torch.randn_like(self.Layer1.weight)
        self.L1WLogStd = std_scale*torch.ones_like(self.Layer1.weight)
        self.L1WStd = torch.exp(self.L1WLogStd)

        self.L1BMean = 0*torch.randn_like(self.Layer1.bias)
        self.L1BLogStd = std_scale*torch.ones_like(self.Layer1.bias)
        self.L1BStd = torch.exp(self.L1BLogStd)


        self.L2WMean = 0*torch.randn_like(self.Layer2.weight)
        self.L2WLogStd = std_scale*torch.ones_like(self.Layer2.weight)
        self.L2WStd = torch.exp(self.L2WLogStd)

        self.L2BMean = 0*torch.randn_like(self.Layer2.bias)
        self.L2BLogStd = std_scale*torch.ones_like(self.Layer2.bias)
        self.L2BStd = torch.exp(self.L2BLogStd)
        
        #Declare layer weights and biases as Pyro samples from normal distribution based on previously-declared Pyro parameters
        self.Layer1.weight = PyroSample(dist.Normal(self.L1WMean, self.L1WStd).independent(2))
        self.Layer1.bias = PyroSample(dist.Normal(self.L1BMean, self.L1BStd).independent(1))

        self.Layer2.weight = PyroSample(dist.Normal(self.L2WMean, self.L2WStd).independent(2))
        self.Layer2.bias = PyroSample(dist.Normal(self.L2BMean, self.L2BStd).independent(1))

    def forward(self, input_data):
        #Simply runs through the network stack and returns the outputs

        input_data = self.flatten(input_data)
        hidden_data = nn.functional.tanh(self.Layer1(input_data))
        output_data = self.Layer2(hidden_data)
        
        return output_data

#-----NEURAL NETWORK INITIALISATION-----#
#Declare whether to sample properties based on their uncertainties, or just include the uncertainties as unique features
sampled_uncertainties = False

#Define network node sizes, and create a neural network instance for classification plus one for redshift predictions
prior_scale_class = -1 #Determines the size of the Gaussian priors for weights and biases
prior_scale_redshift = 0

global_redshift_noise_sampling = True #Declare whether to infer observational noise in redshifts (True) or treat as a fixed value (False)
global_redshift_obs_noise_scale = -5 #Prior for observation noise modelled into predictions - very low value treats redshift outputs as a point-prediction - represents irreducible uncertainty in the redshift predictions

data_imputation = False #Declare whether to impute missing data to boost the size of the dataset, or whether to use 'clean data' only!

including_classifications_in_training = False

#####sns.heatmap(np.corrcoef(train_data_array, rowvar=False))#####


'''
#Used when including uncertainties as separate features
if sampled_uncertainties:
    input_nodes_classification = 17
    input_nodes_redshifts = 18 #Same inputs as classification plus the class probability
else:
    input_nodes_classification = 23
    input_nodes_redshifts = 24
'''

#Not including uncertainties as separate features
#Either use them in sampling (sampled_uncertainties = True) or don't use them at all (sampled_uncertainties = False)
if including_classifications_in_training:
    input_nodes_classification = 17
    input_nodes_redshifts = 18 #Same inputs as classifier plus the class probability
else:
    input_nodes_classification = 17
    input_nodes_redshifts = 17 #Same inputs as classifier


hidden_nodes_classification = 32
hidden_nodes_redshifts = 4

output_nodes_classification = 2
output_nodes_redshifts = 1

if __name__=="__main__":
    print("Using feature values sampled based on uncertainties:", sampled_uncertainties)
    print(f"Using {hidden_nodes_classification} hidden nodes for classifier and a log prior width of {prior_scale_class}.\nUsing {hidden_nodes_redshifts} hidden nodes for redshift predictor and a log prior width of {prior_scale_redshift}, and an observation noise scale of {global_redshift_obs_noise_scale}.")
    print("Inferring an observation noise scale:", global_redshift_noise_sampling)
    print("Imputing missing data columns:", data_imputation)

classification_neural_network = BayesianNeuralNetwork(input_nodes_classification, hidden_nodes_classification, output_nodes_classification, prior_scale_class)
redshifts_neural_network = BayesianNeuralNetwork(input_nodes_redshifts, hidden_nodes_redshifts, output_nodes_redshifts, prior_scale_redshift)

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

    input_features_with_samples = UncertaintySampling(input_features, sampled_uncertainties)
    #input_features_with_samples = input_features
    
    #print(input_features_with_samples[7, 12:])
    logits = classification_neural_network(input_features_with_samples)
                
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
    
    input_features_with_samples = UncertaintySampling(input_features, sampled_uncertainties, redshifts=True)
    
    #print(input_features_with_samples[7, 12:])
    outputs = redshifts_neural_network(input_features_with_samples)

    noise_sampling = global_redshift_noise_sampling
    prior_noise_scale = global_redshift_obs_noise_scale #Prior for observation noise modelled into predictions - very low value treats redshift outputs as a point-prediction
    
    
    if noise_sampling:
        noise_level = pyro.sample("log_sigma", dist.Normal(prior_noise_scale, 0.5))
    else:
        noise_level = torch.tensor(prior_noise_scale)
    predictions = outputs.flatten()

    pyro.deterministic("output_redshift", predictions)
    with pyro.plate("results_redshift", outputs.shape[0]):
        pyro.sample("obs_redshift", dist.Normal(predictions, torch.exp(noise_level)), obs=correct_redshifts)
    return predictions
    
    
def UncertaintySampling(input_features, sampling_with_uncertainties, redshifts=False):
    '''
    Properly deals with the uncertainty sampling for the model
    If sampling_with_uncertainties is True, we treat each of the uncertain features as a latent variable and sample a value from a normal dist. scaled by their associated catalog uncertainties
        -> Must do it this way for HMC - random sampling of the input values collapses training to infinitesimal step sizes and 0 acceptance probability!
    If sampling_with_uncertainties is False, we simply train on the fixed mean value for each uncertain feature
    Inputs:
        Tensor containing the input data
        Boolean defining whether we want to sample uncertainties or not
        *Optional* Boolean as to whether we are training with redshifts or not (defaults to False)
    Returns:
        Tensor with the uncertain input feature columns properly filled, and their associated uncertainty columns filtered out
    '''
    if redshifts and including_classifications_in_training:
        non_unc_inputs = [0, 1, 2, 7, 10, 15, 16, 17, 18, 19, 22, 23]
    else:
        non_unc_inputs = [0, 1, 2, 7, 10, 15, 16, 17, 18, 19, 22]
        
    #unc_inputs = [3,4, 5,6, 8,9, 11,12, 13,14, 20,21]

    #Treat features as normal distributions with their uncertainties
    if sampling_with_uncertainties:
        
        with pyro.plate("uncertainties", input_features.shape[0]):
            #Flux1000 = torch.abs(pyro.sample("Flux1000", dist.Normal(input_features[:, 3], input_features[:, 4])))
            #Energy_Flux100 = torch.abs(pyro.sample("Energy_Flux100", dist.Normal(input_features[:, 5], input_features[:, 6])))
            #PL_Index = torch.abs(pyro.sample("PL_Index", dist.Normal(input_features[:, 8], input_features[:, 9])))
            #LP_Index = torch.abs(pyro.sample("LP_Index", dist.Normal(input_features[:, 11], input_features[:, 12])))
            #LP_beta = pyro.sample("LP_beta", dist.Normal(input_features[:, 13], input_features[:, 14])) #Doesn't need to be positive!
            #Frac_Variability = torch.abs(pyro.sample("Frac_Variability", dist.Normal(input_features[:, 20], input_features[:, 21])))
            
            Flux1000 = pyro.sample("Flux1000", dist.Normal(input_features[:, 3], input_features[:, 4]))
            Energy_Flux100 = pyro.sample("Energy_Flux100", dist.Normal(input_features[:, 5], input_features[:, 6]))
            PL_Index = pyro.sample("PL_Index", dist.Normal(input_features[:, 8], input_features[:, 9]))
            LP_Index = pyro.sample("LP_Index", dist.Normal(input_features[:, 11], input_features[:, 12]))
            LP_beta = pyro.sample("LP_beta", dist.Normal(input_features[:, 13], input_features[:, 14])) #Doesn't need to be positive!
            Frac_Variability = pyro.sample("Frac_Variability", dist.Normal(input_features[:, 20], input_features[:, 21]))
        
        #Flux1000 = dist.Normal(input_features[:, 3], input_features[:, 4]).sample()
        #Energy_Flux100 = dist.Normal(input_features[:, 5], input_features[:, 6]).sample()
        #PL_Index = dist.Normal(input_features[:, 8], input_features[:, 9]).sample()
        #LP_Index = dist.Normal(input_features[:, 11], input_features[:, 12]).sample()
        #LP_beta = dist.Normal(input_features[:, 13], input_features[:, 14]).sample() #Doesn't need to be positive!
        #Frac_Variability = dist.Normal(input_features[:, 20], input_features[:, 21]).sample()
    
    #Treat features as fixed, only use their observed "mean" value
    else:
        Flux1000 = input_features[:, 3]
        Energy_Flux100 = input_features[:, 5]
        PL_Index = input_features[:, 8]
        LP_Index = input_features[:, 11]
        LP_beta = input_features[:, 13]
        Frac_Variability = input_features[:, 20]
        
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


##### TRAINING DATA CREATION #####

#Import catalog and filter out missing data
fits_file = fits.open("table-4LAC-DR3-h.fits")
hdu_table = fits_file[1]

if data_imputation:
    master_headers_array, master_data_array = MissingDataImputation(hdu_table)
    if __name__ == "__main__":
        print("Imputed dataset size:", master_data_array.shape)
    
else:
    master_headers_array, master_data_array = MissingDataFiltering(hdu_table)
    if __name__ == "__main__":    
        print("Filtered dataset size:", master_data_array.shape)
    
#Format the string-based columns into network-readable numerical values
master_data_array = PowerLawFormatting(master_data_array)
master_data_array = SEDClassFormatting(master_data_array)

#Filter out non-relevant classifications for training set (FSRQs and BL Lacs)
temp_data_array = ClassificationFiltering(master_data_array)

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

#Convert data arrays into Torch tensors and initialise DataLoaders for classification training
train_data_tensor, train_class_tensor, train_dataloader = InitialiseDataLoaders(train_data_array, train_class_array, batch_size=32)
#val_data_tensor, val_class_tensor, val_dataloader = InitialiseDataLoaders(val_data_array, val_class_array)
test_data_tensor, test_class_tensor, test_dataloader = InitialiseDataLoaders(test_data_array, test_class_array)


##### NEURAL NETWORK TRAINING FUNCTIONS AND PROCESS ######

def MCMCMethod(train_data, train_labels, num_samples, warmup_steps, ModelFunc, num_chains=1):
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
    target_accept_prob = 0.9 #Default is 0.8; increasing it decreases step size within NUTS, making sampling more stable (or "robust", according to Pyro documentation)
    using_full_mass = False
    print("Using a target acceptance probability of", target_accept_prob, "\nUsing full-mass matrix:", using_full_mass)
    
    nuts_kernel = pyro.infer.NUTS(ModelFunc, jit_compile=True, target_accept_prob=target_accept_prob, full_mass=using_full_mass)

    if num_chains == 1:
        mcmc = pyro.infer.mcmc.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    else:
        mcmc = pyro.infer.mcmc.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, mp_context="spawn")#, warmup_steps=min(num_samples, 100))
    
    mcmc.run(train_data, train_labels)
    
    return mcmc

def CLASSIFIERKFOLDCVRESULTS():
    #Raw results gathered from classifier training; 5-fold CV, 2k samples, log prior -1, no input uncertainties modelled
    accuracies = [[89.47368421, 94.15204678, 94.73684211, 91.81286550, 92.94117647],
                  [87.71929825, 93.56725146, 93.56725146, 94.73684211, 94.11764706],
                  [90.64327485, 93.56725146, 93.56725146, 94.73684211, 94.70588235],
                  [90.05847953, 94.73684211, 94.73684211, 94.73684211, 94.70588235],
                  [90.64327485, 94.73684211, 95.90643275, 95.90643275, 95.88235294],
                  [87.71929825, 98.24561404, 98.24561404, 98.24561404, 97.64705882],
                  [88.88888889, 98.83040936, 98.24561404, 98.83040936, 99.41176471],
                  [91.22807018, 99.41520468, 98.83040936, 100.0000000, 99.41176471]]
    
    f1_scores = [[0.83636364, 0.91228070, 0.91588785, 0.87037037, 0.88235294],
                 [0.81081081, 0.89908257, 0.89523810, 0.91743119, 0.90384615],
                 [0.85964912, 0.89908257, 0.89719626, 0.91588785, 0.91588785],
                 [0.83809524, 0.91743119, 0.91588785, 0.91428571, 0.91428571],
                 [0.85185185, 0.92035398, 0.93457944, 0.93457944, 0.93333333],
                 [0.80373832, 0.97247706, 0.97196262, 0.97196262, 0.96226415],
                 [0.82882883, 0.98181818, 0.97196262, 0.98148148, 0.99047619],
                 [0.85981308, 0.99082569, 0.98148148, 1.00000000, 0.99047619]]

    brier_scores = [[0.08036278, 0.04617873, 0.04800424, 0.04769535, 0.05950984],
                    [0.08388846, 0.04540980, 0.04849856, 0.03879308, 0.05012128],
                    [0.07170403, 0.04891137, 0.04592301, 0.04093999, 0.04722300],
                    [0.07035825, 0.04056616, 0.04306242, 0.03714736, 0.03950316],
                    [0.07214631, 0.04093535, 0.03390736, 0.03106765, 0.03417957],
                    [0.07853300, 0.01656798, 0.02066911, 0.01899046, 0.02128244],
                    [0.08179004, 0.01301220, 0.01656851, 0.01298270, 0.01177432],
                    [0.07224519, 0.00836976, 0.01036910, 0.00614519, 0.00656602]]

    auroc = [[0.94444444, 0.97863248, 0.96312124, 0.98338082, 0.96113530],
             [0.95140867, 0.98116493, 0.97404242, 0.98923710, 0.97452024],
             [0.95077556, 0.97831592, 0.97657487, 0.98828743, 0.97839058],
             [0.96169674, 0.98528015, 0.97625831, 0.98907882, 0.98274472],
             [0.95773979, 0.98860399, 0.98385565, 0.99382716, 0.99096920],
             [0.95773979, 0.99905033, 0.99730928, 0.99810066, 0.99725851],
             [0.95030073, 0.99968344, 0.99873378, 0.99952517, 0.99919368],
             [0.96138018, 1.00000000, 0.99952517, 1.00000000, 1.00000000]]
    
    nlpds = [[0.27685100, 0.17856619, 0.19718619, 0.17227516, 0.22351341],
             [0.26926100, 0.16187026, 0.17688330, 0.13870101, 0.18044038],
             [0.25040227, 0.17498344, 0.17168881, 0.14494283, 0.17183070],
             [0.23217957, 0.14551605, 0.16714524, 0.12894349, 0.14476384],
             [0.24476530, 0.13883103, 0.13922618, 0.11183362, 0.12075637],
             [0.26282850, 0.06964295, 0.08339854, 0.07703793, 0.08146767],
             [0.27532125, 0.05891974, 0.07081387, 0.06048841, 0.05612111],
             [0.23814128, 0.04477951, 0.04617491, 0.03795325, 0.03754791]]
    
    nodes_to_test = [2, 4, 8, 16, 32, 64, 128, 256]
    
    
    #Average accuracy from each fold:  [90.64327485 94.73684211 95.90643275 95.90643275 95.88235294] 
    #Average accuracy over all folds:  94.61506707946336
    #Average F1-score from each fold:  [0.85185185 0.92035398 0.93457944 0.93457944 0.93333333] 
    #Average F1-score over all folds:  0.9149396091981487
    #Average Brier score from each fold:  [0.07214631 0.04093535 0.03390736 0.03106765 0.03417957] 
    #Average Brier score over all folds:  0.04244724695343734
    #Average AUC from each fold:  [0.95773979 0.98860399 0.98385565 0.99382716 0.9909692 ] 
    #Average AUC over all folds:  0.982999157841925
    #Average NLPD from each fold:  [0.2447653  0.13883103 0.13922618 0.11183362 0.12075637] 
    #Average NLPD over all folds:  0.15108250081539154

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    ax1.errorbar(np.log2(nodes_to_test), np.mean(accuracies, axis=1), yerr=np.std(accuracies, axis=1), color='red', fmt='x')
    fig.suptitle("Mean accuracy rate, AUROC, Brier Score and NLPD vs. $log_2$ of hidden node number for 5-fold CV")
    ax1.set_ylabel("Accuracy / %")
    
    ax2.errorbar(np.log2(nodes_to_test), np.mean(auroc, axis=1), yerr=np.std(auroc, axis=1), color='green', fmt='x')
    ax2.set_ylabel("AUROC")
    
    ax3.errorbar(np.log2(nodes_to_test), np.mean(brier_scores, axis=1), yerr=np.std(brier_scores, axis=1), color='blue', fmt='x')
    ax3.set_ylabel("Brier Score")
    
    ax4.errorbar(np.log2(nodes_to_test), np.mean(nlpds, axis=1), yerr=np.std(nlpds, axis=1), color='purple', fmt='x')
    ax4.set_ylabel("NLPD")
    
    ax4.set_xlabel("$Log_2$(hidden nodes)")
    plt.show()


def ClassMCMCAccuracy(test_data, test_classes, samples_no_unc, guide, return_metrics, print_values=True, run_number=10):
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


    if samples_no_unc == None:
        num_samples = 500
        predictive = pyro.infer.Predictive(model=SVI_Model_Class, guide=guide, num_samples=num_samples, return_sites={"obs_class", "probabilities"})

    elif guide == None:
        predictive = pyro.infer.Predictive(model=HMC_Model_Class, posterior_samples=samples_no_unc, return_sites={"obs_class", "probabilities"})

    else:
        sys.exit("DID NOT PASS IN A GUIDE OR SAMPLES TO THE REDSHIFT PREDICTIONS FUNCTION!")
    
    accuracy_rates = np.zeros(run_number)
    f1_scores = np.zeros(run_number)
    brier_scores = np.zeros(run_number)
    auc_scores=np.zeros(run_number)

    for i in range(run_number):        
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
    
    if print_values:
        print("Mean accuracy rate for sample set:", np.mean(accuracy_rates), "%")
        print("Mean F1 score for sample set:", np.mean(f1_scores))
        print("Mean Brier score for sample set:", np.mean(brier_scores))
        print("Mean AUC score for sample set:", np.mean(auc_total))

    return preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores

def ClassMCMCPlotting(test_data, test_classes, samples=None, guide=None, plots=True, return_metrics=False):
    '''
    Produces some plots using the MCMC data (e.g: predicted labels histogram, trace plots of Layer 2 weights, entropy of predictions)
    Inputs:
        MCMC method object
        Tensor containing the test data
        Tensor containing the test labels
        *Optional* Dictionary containing the MCMC samples if already obtained - just used to pass through to the ClassMCMCAccuracy function (defaults to None if samples not provided)
    '''    
    preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores = ClassMCMCAccuracy(test_data, test_classes, samples, guide, return_metrics=return_metrics)
    
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
        plt.ylabel("Number of objects")
        plt.xlabel("Entropy")
        plt.ylim([0, 65])
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
        
        plt.scatter(np.mean(probs[:, fsrqs, 1], axis=0), np.std(probs[:, fsrqs, 1], axis=0), color='red', marker='x', alpha=0.8, label='True FSRQs')
        plt.scatter(np.mean(probs[:, blls, 1], axis=0), np.std(probs[:, blls, 1], axis=0), color='blue', marker='x', alpha=0.8, label='True BLLs')
        plt.title("Uncertainty (standard deviation) in FSRQ probability against mean FSRQ probability per object")
        plt.xlabel("Mean FSRQ probability")
        plt.ylabel("Uncertainty in FSRQ probability")
        plt.xlim([-0.1, 1.1])
        plt.ylim([0, 0.40])
        plt.legend()
        plt.show()
        

        prob_true, prob_pred = sklearn.calibration.calibration_curve(test_classes, mean_probs[:, 1], n_bins=10)
        fig = plt.figure()
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], ls='--', label='Perfect calibration', color='black')
        ax.plot(prob_pred, prob_true, label='Classifier model')
        ax.set_xlabel("Average predicted probability")
        ax.set_ylabel("True Positive Rate")
        plt.legend()
        plt.title(f"Calibration curve for {hidden_nodes_classification} node classifier")
        plt.show()
        
        '''
        plt.scatter(np.mean(probs[:, :, 0], axis=0), np.std(probs[:, :, 0], axis=0), color='blue', marker='x')
        plt.title("Uncertainty (standard deviation) in BLL probability against mean BLL probability per object")
        plt.xlabel("Mean BLL probability")
        plt.ylabel("Uncertainty in BLL probability")
        plt.show()
        '''
        #plt.scatter(mean_pred, test_data[:, 14]) #PL_Index against uncertainty
        
    fsrq_log_probs = np.log(mean_probs[fsrqs, 1]).squeeze(0)
    bll_log_probs = np.log(mean_probs[blls, 0]).squeeze(0)
            
    log_pred_density = np.sum(fsrq_log_probs, axis=0) + np.sum(bll_log_probs, axis=0)
    print("Total Log Predictive Density:", log_pred_density,"\nMean LPD:", log_pred_density/test_classes.shape[0])
    print("Min probs. involved:", fsrq_log_probs.min(), bll_log_probs.min())
        
    if return_metrics:
        return preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores, log_pred_density/test_classes.shape[0]
    
    
    
def ClassMCMCPlottingCopy(test_classes, probs_unc, preds_not_unc, probs_not_unc, plots=True, return_metrics=False):
    '''
    Produces some plots using the MCMC data (e.g: predicted labels histogram, trace plots of Layer 2 weights, entropy of predictions)
    Inputs:
        MCMC method object
        Tensor containing the test data
        Tensor containing the test labels
        *Optional* Dictionary containing the MCMC samples if already obtained - just used to pass through to the ClassMCMCAccuracy function (defaults to None if samples not provided)
    '''    

    fsrqs = np.where(test_classes.cpu().numpy() == 1)
    blls = np.where(test_classes.cpu().numpy() == 0)
    
    #mean_probs = np.mean(probs, axis=0)
    #mean_pred = np.argmax(mean_probs, axis=1)
            
    '''    
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
    '''
    
    mean_probs_unc = np.mean(probs_unc, axis=0)
    mean_probs_no_unc = np.mean(probs_not_unc, axis=0)
    
    entropy_unc = -((mean_probs_unc[:, 0]*np.log(mean_probs_unc[:, 0]+1e-9)) + ((mean_probs_unc[:, 1])*np.log(mean_probs_unc[:, 1]+1e-9)))
    entropy_no_unc = -((mean_probs_no_unc[:, 0]*np.log(mean_probs_no_unc[:, 0]+1e-9)) + ((mean_probs_no_unc[:, 1])*np.log(mean_probs_no_unc[:, 1]+1e-9)))
    
    fig = plt.figure()
    ax_e = fig.gca()
    ax_e.hist(entropy_no_unc, bins=40, alpha=0.4, label='No unc. model')
    ax_e.hist(entropy_unc, bins=40, alpha=0.4, label='Unc. model')
    ax_e.set_title("Entropy of each object prediction")
    ax_e.set_ylabel("Number of objects")
    ax_e.set_xlabel("Entropy")
    ax_e.set_ylim([0, 65])
    ax_e.legend()
        
    fig = plt.figure()
    ax_s = fig.gca()
    ax_s.scatter(np.mean(probs_not_unc[:, fsrqs, 1], axis=0), np.std(probs_not_unc[:, fsrqs, 1], axis=0), color='pink', marker='x', alpha=0.8, label='True FSRQs - no unc. model')
    ax_s.scatter(np.mean(probs_not_unc[:, blls, 1], axis=0), np.std(probs_not_unc[:, blls, 1], axis=0), color='lightblue', marker='x', alpha=0.8, label='True BLLs - no unc. model')
    ax_s.scatter(np.mean(probs_unc[:, fsrqs, 1], axis=0), np.std(probs_unc[:, fsrqs, 1], axis=0), color='red', marker='x', alpha=0.8, label='True FSRQs - unc. model')
    ax_s.scatter(np.mean(probs_unc[:, blls, 1], axis=0), np.std(probs_unc[:, blls, 1], axis=0), color='blue', marker='x', alpha=0.8, label='True BLLs - unc. model')
    ax_s.set_title("Uncertainty (standard deviation) in FSRQ probability against mean FSRQ probability per object")
    ax_s.set_xlabel("Mean FSRQ probability")
    ax_s.set_ylabel("Uncertainty in FSRQ probability")
    ax_s.set_xlim([-0.1, 1.1])
    ax_s.set_ylim([0, 0.40])
    ax_s.legend()
    
    plt.show()
        


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
            
        #mcmc = MCMCMethod(temp_train_data, temp_train_classes, num_samples_class, num_samples_class, ClassificationModelFunc)
        #list_mcmcs_class.append(mcmc)
        preds, probs, accuracies, f1_scores, brier_scores, auc_scores, lpd = ClassMCMCPlotting(temp_val_data, temp_val_classes, samples=mcmc.get_samples(), return_metrics=True, plots=False)
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
    Used when importing cross-validation sample dictionaries that have already been merged
    Inputs: 
        Dictionary containing all samples (and keys)
    Returns: 
        List of dictionaries containing the samples for the network weights and biases for each fold
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
    '''
    Filters the master AGN array by class for BCU sources
    Inputs:
        Array containing all of the AGN data
    Returns:
        Array containing only the BCU sources
    '''
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

def ClassifyingBCUs(input_master_array, sample_set=None, guide=None, zscore_means=zscore_means, zscore_stds=zscore_stds):
    '''
    Runs the data reduction of BCU source information, passing these values into the classification network to obtain class probabilities
    Inputs:
        Array containing all of the AGN data
        Dictionary of the classification network samples to be used in generating predictions
        Float array for the z-score means (for data normalisation)
        Float array for the z-score stds (for data normalisation)
    Returns:
        Tensor containing the redshift training data for the BCUs
        Array containing the class probabilities for the BCUs
        Array containing the redshifts for the BCUs
        Array containing the source IDs for the BCUs
    '''
    master_bcu_array = RetrieveBCUs(input_master_array)
    
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
    
    if sample_set == None:
        print("No sample set provided to BCU function - importing a default one!")
        samples_file_name = 'temp_samples_dict.npy'
        sample_set = LoadSamples(samples_file_name)
    
    if using_HMC:
        predictive = pyro.infer.Predictive(model=HMC_Model_Class, posterior_samples=sample_set, return_sites={"obs_class", "probabilities"})
    elif using_SVI:
        predictive = pyro.infer.Predictive(model=SVI_Model_Class, guide=guide, num_samples=500, return_sites={"obs_class", "probabilities"})

    output = predictive(bcu_data_tensor)
    probs = output['probabilities'].squeeze().cpu().numpy()
    
    #ClassMCMCAccuracy(bcu_data_tensor, test_classes, samples_no_unc=sample_set, guide=guide)
    
    return bcu_data_tensor, probs, bcu_redshift_array, bcu_source_name_array

def ListMergingAndTrimmingMCMCs(mcmc_list, is_redshifts):
    '''
    Creates new reduced sample dictionaries containing only the network weights/biases, and observation noise (log_sigma) if relevant
    Inputs:
        List of the MCMC objects containing the samples 
        Boolean of whether this is a redshifts sample dictionary or not (defines whether to include log_sigma or not)
    Returns:
        Dictionary of all requested MCMC samples merged into one
    '''
    #Trim out unnecessary columns and combine the MCMC samples from each fold into one large dictionary    
    all_samples = {}
    if is_redshifts: 
        network_keys = ['Layer1.weight', 'Layer1.bias', 'Layer2.weight', 'Layer2.bias', 'log_sigma']
    else: 
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
    
def DataFormattingForRedshifts(input_data_tensor, input_class_tensor, input_redshift_array, input_source_name_array, class_samples, include_classifications=True):
    '''
    Formats the input data tensor to make it ready for input into redshift model - only using objects with known redshifts for training!
    If classifications are wanted in the redshift input data, generates class probabilities for the objects and adds these to the input data tensor
    Inputs:
        Tensor containing the input data for classification
        Tensor containing the objects' class labels
        Array containing the objects' true redshift values
        Array containing the objects' source names
        Dictionary containing the samples of the trained classification model
        *Optional* Boolean for whether to include classification probabilities in the training data
    Returns:
        Tensor containing the input data for redshifts
        Tensor containing the known redshifts of the sources
        Array containing the source names of the objects 
    '''    
    if include_classifications:
        preds, probs = ClassMCMCAccuracy(input_data_tensor, input_class_tensor, class_samples, print_values=False, return_metrics=False, run_number=1)
        mean_probs_tensor = torch.tensor(np.mean(probs[:, :, 1], axis=0)) #Taking FSRQ probabilities as inputs
        temp_data_tensor = torch.cat((input_data_tensor, mean_probs_tensor.unsqueeze(1)), dim=1)
        
    else:
        temp_data_tensor = input_data_tensor

    known_redshift_indices = np.where(input_redshift_array != -np.inf)[0]
    data_tensor_for_redshifts = temp_data_tensor[known_redshift_indices]
    known_redshifts_tensor = torch.tensor(input_redshift_array[known_redshift_indices])
    redshift_source_names_array = input_source_name_array[known_redshift_indices]
    
    return data_tensor_for_redshifts, known_redshifts_tensor, redshift_source_names_array

def RedshiftPerformance(input_data_tensor, input_redshift_tensor, samples=None, guide=None, redshifts_z_scored=(), plots=False, return_metrics=False, prints=True):
    '''
    Produces diagnostic information on the performance of the sample set/guide when applied to the input dataset    
    Inputs:
        Tensor containing the input data for the model
        Tensor containing the correct redshift values
        Dictionary containing the samples for the model obtained from HMC
        *Optional* Tuple containing the mean and std. to reverse the z-scoring for the redshift values for plotting
        *Optional* Boolean for whether to generate plots
        *Optional* Boolean for whether to return generated outputs (no obs noise modelled) and predictions (with obs noise modelled)
    Outputs:
        *Optional* Arrays containing the generated outputs and predictions from the model 
        *Optional* Float numbers corresponding to some diagnostic/performance measures of the model
    '''
    if guide == None:
        method = "HMC method"
    
    elif samples == None:
        method = "SVI method"
        
    else:
        sys.exit("DID NOT PASS IN A GUIDE OR SAMPLES TO THE REDSHIFT PLOTTING FUNCTION!")
    
    input_redshift_array = input_redshift_tensor.cpu().numpy()
    
    redshift_outputs_tensor, redshift_preds_tensor, log_sigma_tensor = RedshiftPredictions(input_data_tensor, samples, guide)
    redshift_outputs = redshift_outputs_tensor.cpu().numpy().squeeze()
    redshift_preds = redshift_preds_tensor.cpu().numpy().squeeze()

    #Reverses the z-scoring of the redshift values if it was done before training
    if len(redshifts_z_scored) == 2:
        (redshift_mean, redshift_std) = redshifts_z_scored
        redshift_mean, redshift_std = redshift_mean.cpu().numpy(), redshift_std.cpu().numpy()
        input_redshift_tensor = (input_redshift_tensor*redshift_std) + redshift_mean
        redshift_outputs = (redshift_outputs * redshift_std) + redshift_mean
        redshift_preds = (redshift_preds*redshift_std) + redshift_mean
    avg_redshift_preds = np.mean(redshift_preds, axis=0)
    var_redshift_preds = np.var(redshift_preds, axis=0)
    
    var_redshift_outputs = np.var(redshift_outputs, axis=0)    
    
    average_variance_from_obs_noise = torch.mean(torch.exp(log_sigma_tensor)**2)
    
    redshifts_good_fit_indices = np.where(var_redshift_preds < 16)[0] #Tends to be higher prediction (observation) noise than output noise - use this as a filter
    avg_redshift_preds = avg_redshift_preds[redshifts_good_fit_indices]
    var_redshift_preds = var_redshift_preds[redshifts_good_fit_indices]
    var_redshift_outputs = var_redshift_outputs[redshifts_good_fit_indices]
    
    input_redshift_tensor = input_redshift_tensor[redshifts_good_fit_indices]
    input_redshift_array = input_redshift_tensor.cpu().numpy()
    
    RMSE = np.sqrt(np.sum(((avg_redshift_preds - input_redshift_array)**2)/input_redshift_array)/len(input_redshift_array))
    
    RMSE_true = np.sqrt(np.sum(((avg_redshift_preds - input_redshift_array)**2))/len(input_redshift_array))
    
    chi_squared_reduced = np.sum(((avg_redshift_preds - input_redshift_array)**2)/var_redshift_preds)/(len(input_redshift_array)-input_nodes_redshifts)
    
    def LeastSquaresFitting(predicted_data, actual_data):
        N = actual_data.shape[0]
        delta = N * np.sum(actual_data**2) - (np.sum(actual_data))**2
        gradient = (N * np.sum(predicted_data*actual_data) - np.sum(predicted_data) * np.sum(actual_data))/delta
        intercept = (np.sum(actual_data**2)*np.sum(predicted_data) - np.sum(actual_data)*np.sum(actual_data * predicted_data))/delta

        acu = np.sqrt(1/(N-2) * np.sum((predicted_data - gradient * actual_data - intercept)**2))

        err_gradient = acu * (N/delta)
        err_intercept = acu * np.sqrt(np.sum(actual_data**2)/delta)
        
        return gradient, intercept, err_gradient, err_intercept
    
    #Predicted vs. Actual line of best fit
    gradient, intercept, err_gradient, err_intercept = LeastSquaresFitting(avg_redshift_preds, input_redshift_array)

    #R^2 calculation
    r_squared = 1 - (np.sum((avg_redshift_preds - input_redshift_array)**2)/np.sum((input_redshift_array - np.mean(input_redshift_array))**2))
    #Log Predictive Density calculations
    normal_dist = dist.Normal(redshift_outputs_tensor.squeeze(1)[:, redshifts_good_fit_indices], torch.exp(log_sigma_tensor))
    log_probs = normal_dist.log_prob(input_redshift_tensor.unsqueeze(0))
    log_pred_density = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(log_probs.shape[0], dtype=log_probs.dtype))
    mean_lpd = log_pred_density.mean()
    #mean_lpd = 2
    
    #Residuals line of best fit
    residuals = avg_redshift_preds-input_redshift_array
    res_grad, res_intercept, res_err_grad, res_error_intercept = LeastSquaresFitting(residuals, input_redshift_array)

    if prints:
        print("Mean log_sigma value:", log_sigma_tensor.mean())
        print("Mean obs. sigma value:", torch.mean(torch.exp(log_sigma_tensor)))
        print("Outliers in predictions and outputs respectively:", np.where(var_redshift_preds >= 16)[0], np.where(var_redshift_outputs > 4)[0])
        print("Average std. of OUTPUT redshifts without outliers (std. < 4):", np.sqrt(np.mean(var_redshift_outputs)))
        print("Average std. of OBSERVED redshifts without outliers (std. < 4):", np.sqrt(np.mean(var_redshift_preds)))
        print("Weighted (legacy) RMSE:", RMSE)
        print("Proper RMSE:", RMSE_true)
        print("Reduced chi-squared:", chi_squared_reduced)
        print("R-squared:", r_squared)
        print("Mean LPD:", mean_lpd.item())
        print("Max. true redshift value:", max(input_redshift_array))

    '''
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(wspace=0, hspace=0.05)
    
    for x in [5, 1000, 3000]:
        ax1.scatter(input_redshift_array, redshift_preds[x][redshifts_good_fit_indices], marker='x', label=f'Sample set {x}')
        ax1.plot([0, 3], [0, 3], ls='--', color='black', label='Perfect prediction accuracy')
        ax2.scatter(input_redshift_array, redshift_preds[x][redshifts_good_fit_indices] - input_redshift_array, marker='x')
        RMSE = np.sqrt(np.sum(((redshift_preds[x][redshifts_good_fit_indices] - input_redshift_array)**2))/len(input_redshift_array))
        print(f"Sample {x} RMSE:", RMSE)
        ax1.set_xlim([0, 3])
        ax1.set_ylim([0, 3])
        ax1.set_ylabel("Predicted redshift values")
        ax1.set_title("Scatter plot of predicted values from individual sample sets against actual values of redshift on the test dataset")
        ax1.legend()
        ax2.plot([0, 3], [0, 0], ls='--', lw=1, color='black')
        ax2.set_ylabel("Residual")
        ax2.set_ylim([-2, 2])
        ax2.set_xlabel("Actual redshift values")
    '''
    
    #fsrqs = np.where(test_classes.cpu().numpy() == 1)
    #blls = np.where(test_classes.cpu().numpy() == 0)
    
    if plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(avg_redshift_preds, density=True, alpha=0.6, label="Predicted", bins=20)
        ax.hist(input_redshift_array, density=True, alpha=0.6, label="Actual", bins=20)
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title("Histogram of the predicted values and actual values of redshift for the test dataset")
        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8,6))
        plt.subplots_adjust(wspace=0, hspace=0.05)
        ax1.errorbar(input_redshift_array, avg_redshift_preds, yerr=np.sqrt(var_redshift_preds), fmt='x')

        #ax1.errorbar(input_redshift_array[fsrqs], avg_redshift_preds[fsrqs], yerr=np.sqrt(var_redshift_preds[fsrqs]), fmt='x', color='red', label='FSRQs')
        #ax1.errorbar(input_redshift_array[blls], avg_redshift_preds[blls], yerr=np.sqrt(var_redshift_preds[blls]), fmt='x', color='red', label='BLLs')

        max_redshift_value = max(np.max(input_redshift_array), np.max(avg_redshift_preds), 3)
        #min_redshift_value = max(np.min(input_redshift_array), np.min(avg_redshift_preds))

        max_redshift_value += 0.1

        ax1.plot([0, max_redshift_value], [0, max_redshift_value], ls='--', color='black', label='Perfect prediction accuracy')
        ax1.plot([0, max_redshift_value], [intercept, max_redshift_value*gradient + intercept], color='red', ls='--', label=f"Linear line-of-best-fit: y = {gradient:.3f}x+{intercept:.3f}")
        ax1.plot([0, max_redshift_value], [intercept+err_intercept, max_redshift_value*(gradient+err_gradient) + (intercept+err_intercept)], ls='--', lw=0.5, color='black')
        ax1.plot([0, max_redshift_value], [intercept-err_intercept, max_redshift_value*(gradient-err_gradient) + (intercept-err_intercept)], ls='--', lw=0.5, color='black')
        ax1.fill_between([0, max_redshift_value], [intercept-err_intercept, max_redshift_value*(gradient-err_gradient) + (intercept-err_intercept)], [intercept+err_intercept, max_redshift_value*(gradient+err_gradient) + (intercept+err_intercept)], color='gray', alpha=0.6, label='Error in linear fit')
        ax1.set_xlim([0, max_redshift_value])
        ax1.set_ylim([0, max_redshift_value])
        ax1.set_ylabel("Predicted redshift values")
        ax1.set_title(f"Mean predicted values against actual values of redshift for the test dataset - {method}")
        ax1.legend()
        ax2.plot([0, max_redshift_value], [0, 0], ls='--', lw=1, color='black')
        ax2.scatter(input_redshift_array, avg_redshift_preds-input_redshift_array, marker='x', color='black')
        ax2.plot([0, max_redshift_value], [res_intercept, max_redshift_value*res_grad + res_intercept], color='red', ls='--', lw=0.8, label=f'y = {res_grad:.3f}x + {res_intercept:.3f}')
        ax2.set_ylabel("Residual")
        ax2.set_ylim([-(residuals.max()), residuals.max()])
        ax2.set_xlabel("Actual redshift values")
        ax2.legend()
        
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(residuals, color='black', alpha=0.8, density=True, bins=20)
        ax.set_title("Residuals Histogram")
        ax.set_xlabel("Residual size")
        ax.set_ylabel("Density")
        
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(np.sqrt(var_redshift_preds), np.abs(residuals), marker='x', color='black')
        ax.set_ylabel("Residual")
        ax.set_xlabel("Redshift std.")
        ax.set_title("Plot of the error against the uncertainty in redshift predictions")
        
        ############CORRECTING FOR RESIDUALS TREND PLOT#############        
        corr_avg_redshift_preds = avg_redshift_preds - (input_redshift_array*res_grad + res_intercept)
        corr_grad, corr_intercept, corr_err_grad, corr_err_intercept = LeastSquaresFitting(corr_avg_redshift_preds, input_redshift_array)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8,5))
        plt.subplots_adjust(wspace=0, hspace=0.05)
        ax1.errorbar(input_redshift_array, corr_avg_redshift_preds, yerr=np.sqrt(var_redshift_preds), fmt='x')#, marker='x')
        ax1.plot([0, max_redshift_value], [0, max_redshift_value], ls='--', color='black', label='Perfect prediction accuracy')
        ax1.plot([0, max_redshift_value], [corr_intercept, max_redshift_value*corr_grad + corr_intercept], color='red', ls='--', label=f"Linear line-of-best-fit: y = {corr_grad:.3f}x+{corr_intercept:.3f}")
        ax1.plot([0, max_redshift_value], [corr_intercept+corr_err_intercept, max_redshift_value*(corr_grad+corr_err_grad) + (corr_intercept+corr_err_intercept)], ls='--', lw=0.5, color='black')
        ax1.plot([0, max_redshift_value], [corr_intercept-corr_err_intercept, max_redshift_value*(corr_grad-corr_err_grad) + (corr_intercept-corr_err_intercept)], ls='--', lw=0.5, color='black')
        ax1.fill_between([0, max_redshift_value], [corr_intercept-corr_err_intercept, max_redshift_value*(corr_grad-corr_err_grad) + (corr_intercept-corr_err_intercept)], [corr_intercept+corr_err_intercept, max_redshift_value*(corr_grad+corr_err_grad) + (corr_intercept+corr_err_intercept)], color='gray', alpha=0.6, label='Error in linear fit')
        ax1.set_xlim([0, max_redshift_value])
        ax1.set_ylim([0, max_redshift_value])
        ax1.set_ylabel("Predicted redshift values")
        ax1.set_title("Predicted vs. actual on test dataset corrected for residuals trend")
        ax1.legend()
        ax2.plot([0, max_redshift_value], [0, 0], ls='--', lw=1, color='black')
        ax2.scatter(input_redshift_array, corr_avg_redshift_preds-input_redshift_array, marker='x', color='black')
        #ax2.plot([0, max_redshift_value], [res_intercept, max_redshift_value*res_grad + res_intercept], color='red', ls='--', lw=0.8)
        ax2.set_ylabel("Residual")
        ax2.set_ylim([-max(max(corr_avg_redshift_preds-input_redshift_array), 2), max(max(corr_avg_redshift_preds-input_redshift_array), 2)])
        ax2.set_xlabel("Actual redshift values")
        
        new_RMSE = np.sqrt(np.sum(((corr_avg_redshift_preds - input_redshift_array)**2)/input_redshift_array)/len(input_redshift_array))
        
        new_RMSE_true = np.sqrt(np.sum(((corr_avg_redshift_preds - input_redshift_array)**2))/len(input_redshift_array))
        
        new_chi_squared_reduced = np.sum(((corr_avg_redshift_preds - input_redshift_array)**2)/np.var(corr_avg_redshift_preds))/(len(input_redshift_array)-input_nodes_redshifts) 
        
        new_r_squared = 1 - (np.sum((corr_avg_redshift_preds - input_redshift_array)**2)/np.sum((input_redshift_array - np.mean(input_redshift_array))**2))
        
        if prints:
            print("APPLYING CORRECTION STUFF HERE:")
            print("Weighted (legacy) RMSE:", new_RMSE)
            print("Proper RMSE:", new_RMSE_true)
            print("Reduced chi-squared:", new_chi_squared_reduced)
            print("Corrected r-squared:", new_r_squared)
        ############################################################
     
        index_to_plot = 27
        fig = plt.figure()
        ax = fig.gca()
        array_temp = ax.hist(redshift_preds[:, index_to_plot], bins=50, density=True, alpha=0.8)
        lower_1_sigma = np.percentile(redshift_preds[:, index_to_plot], 16)
        higher_1_sigma = np.percentile(redshift_preds[:, index_to_plot], 84)
        ax.plot([lower_1_sigma, lower_1_sigma], [0, max(array_temp[0])], color='red', ls='--')
        ax.plot([higher_1_sigma, higher_1_sigma], [0, max(array_temp[0])], color='red', ls='--', label=(r'1$\sigma$ region, width of'+f'{higher_1_sigma-lower_1_sigma:.2f}'))
        ax.plot([input_redshift_array[index_to_plot], input_redshift_array[index_to_plot]], [0, max(array_temp[0])], color='black', ls='--', label='True redshift of source')
        ax.set_xlabel("Predicted redshift")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Histogram of predictions over all network samples for source, index {index_to_plot} in test dataset")

        #fig = plt.figure()
        #ax = fig.gca()
        #temp = ax.boxplot(redshift_preds[:, index_to_plot])
        #print(temp)
        
    if return_metrics:
        return redshift_preds, redshift_outputs, mean_lpd.item(), RMSE_true, r_squared, chi_squared_reduced

def RedshiftPredictions(input_data_tensor, samples, guide):
    '''
    Generates redshift predictions using the provided data tensor and either the sample set from HMC or the guide from SVI
    Inputs:
        Tensor containing the input data
        Dict. containing the HMC samples (or None if SVI was used to train)
        Guide object obtained from SVI training (or None if HMC was used to train)
    '''
    #Creates correct predictive depending on if SVI or HMC was used, and if log_sigma was sampled or fixed
    noise_sampling = global_redshift_noise_sampling
    if guide == None:
        num_samples = samples['Layer2.bias'].shape[0]
        if noise_sampling:
            predictive = pyro.infer.Predictive(model=HMC_Model_Redshift, posterior_samples=samples, return_sites={"obs_redshift", "output_redshift", "log_sigma"})
        else:
            predictive = pyro.infer.Predictive(model=HMC_Model_Redshift, posterior_samples=samples, return_sites={"obs_redshift", "output_redshift"})
        
    elif samples == None:
        num_samples = 500
        if noise_sampling:
            predictive = pyro.infer.Predictive(model=SVI_Model_Redshift, guide=guide, num_samples=num_samples, return_sites={"obs_redshift", "output_redshift", "log_sigma"})
        else:
            predictive = pyro.infer.Predictive(model=SVI_Model_Redshift, guide=guide, num_samples=num_samples, return_sites={"obs_redshift", "output_redshift"})
        
    else:
        sys.exit("DID NOT PASS IN A GUIDE OR SAMPLES TO THE REDSHIFT PREDICTIONS FUNCTION!")
    
    output = predictive(input_data_tensor)
    
    if not noise_sampling:
        print("Noise scale was not sampled during training!")
        output['log_sigma'] = (torch.zeros(num_samples)+global_redshift_obs_noise_scale).squeeze()
        
    return output['output_redshift'], output['obs_redshift'], output['log_sigma']

def FeatureImportanceTestRedshifts(input_data_tensor, input_answer_tensor, samples=None, guide=None, redshifts_z_scored=()):
    '''
    Runs through each feature column in the input data in turn, and shuffles that specific feature
    Then, predictions are re-run on the dataset with only that shuffled feature and diagnostics are logged
    Plots the diagnostic metrics for each feature being shuffled
    This tests the reliance of the model on each feature
        -> Large changes in performance suggest that the model has learned the corresponding feature is of particular importance to predicting redshifts
    Inputs:
        Tensor containing the input data for the neural network
        Tensor containing the correct redshift values
        *Optional* Dict containing HMC samples
        *Optional* Guide object obtained from SVI training
        *Optional* Tuple containing the mean and std. to reverse redshift z-scoring
    Returns:
        Nothing!
    '''
    if including_classifications_in_training:
        non_uncertainty_features = [0, 1, 2, 3, 5, 7, 8, 10, 11, 13, 15, 16, 17, 18, 19, 20, 22, 23]
    else:
        non_uncertainty_features = [0, 1, 2, 3, 5, 7, 8, 10, 11, 13, 15, 16, 17, 18, 19, 20, 22]
    
    #Index 0 will equate to the base model
    master_lpd = np.zeros(len(non_uncertainty_features)+1)
    master_RMSE = np.zeros(len(non_uncertainty_features)+1)
    master_r_squared = np.zeros(len(non_uncertainty_features)+1)
    master_chi_squared_reduced = np.zeros(len(non_uncertainty_features)+1)
    master_avg_pred_std = np.zeros(len(non_uncertainty_features)+1)
    
    master_feature_names = []
    master_feature_names.append("Base Model")
    
    preds, outputs, master_lpd[0], master_RMSE[0], master_r_squared[0], master_chi_squared_reduced[0] = RedshiftPerformance(input_data_tensor, input_answer_tensor, samples=samples, guide=guide, redshifts_z_scored=redshifts_z_scored, plots=False, return_metrics=True, prints=False)
    master_avg_pred_std[0] = np.std(np.mean(preds, axis=0))
    
    count=0
    #print(input_data_tensor[3])
    for i in non_uncertainty_features:
        count+=1
        tensor_with_shuffled_feature = torch.clone(input_data_tensor)
        tensor_with_shuffled_feature[:, i] = tensor_with_shuffled_feature[torch.randperm(input_data_tensor.shape[0]), i]
        
        preds, outputs, master_lpd[count], master_RMSE[count], master_r_squared[count], master_chi_squared_reduced[count] = RedshiftPerformance(tensor_with_shuffled_feature, input_answer_tensor, samples=samples, guide=guide, redshifts_z_scored=redshifts_z_scored, plots=False, return_metrics=True, prints=False)
        master_avg_pred_std[count] = np.std(np.mean(preds, axis=0))
        master_feature_names.append(master_headers_array[features_master_list[i]])
        '''
        print("\nFeature ", master_headers_array[features_master_list[i]])
        print("LPD:", master_lpd[i])
        print("RMSE:", master_RMSE[i])
        print("R^2:", master_r_squared[i])
        print("Reduced chi-squared:", master_chi_squared_reduced[i])
        print("Avg. std.:", master_avg_pred_std[i])
        '''
    x_indices = np.linspace(1, len(non_uncertainty_features)+1, len(non_uncertainty_features)+1)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(8,12))
    plt.subplots_adjust(wspace=0, hspace=0)
    
    ax1.scatter(x_indices, -master_lpd+master_lpd[0], color='purple', marker='x')
    ax1.plot([1, len(non_uncertainty_features)+1], [0,0], color='black', ls='--', label='Base Model Value')
    ax1.set_ylabel(r"$\Delta$NLPD")
    ax1.legend()
    
    ax2.scatter(x_indices, master_RMSE-master_RMSE[0], color='blue', marker='x')
    ax2.plot([1, len(non_uncertainty_features)+1], [0,0], color='black', ls='--')
    ax2.set_ylabel(r"$\Delta$RMSE")
    
    ax3.scatter(x_indices, master_avg_pred_std-master_avg_pred_std[0], color='red', marker='x')
    ax3.plot([1, len(non_uncertainty_features)+1], [0,0], color='black', ls='--')
    ax3.set_ylabel(r"$\Delta$Avg. std.")
    #print(master_feature_names)
    
    ax4.scatter(x_indices, master_r_squared-master_r_squared[0], color='green', marker='x')
    ax4.plot([1, len(non_uncertainty_features)+1], [0,0], color='black', ls='--')
    ax4.set_ylabel(r"$\Delta R^2$")    
    
    ax5.scatter(x_indices, master_chi_squared_reduced-master_chi_squared_reduced[0], color='lightblue', marker='x')
    ax5.plot([1, len(non_uncertainty_features)+1], [0,0], color='black', ls='--')
    ax5.set_ylabel(r"$\Delta\chi^2_{\nu}$")
    ax5.set_xticks(range(1, len(x_indices)+1), master_feature_names, size='small', rotation=90)
    ax5.set_xlabel("Shuffled Feature")
    fig.suptitle("Feature Importance Test Results")
    
    
def SVITrainingLoop(input_dataloader, model):
    '''
    Runs the SVI training loop across the dataloader for a predefined number of epochs
    Inputs:
        DataLoader object of the training dataset
    Returns:
        Learned guide object from training
        List containing the loss values after each SVI step
    '''
    num_epochs=100
    pyro.clear_param_store()
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optimizer = pyro.optim.ClippedAdam({"lr": 1e-4, 'lrd': 0.01**(1/(num_epochs*len(input_dataloader)))})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())
    
    for epoch in range(num_epochs):
        for batch in enumerate(input_dataloader):
            loss = svi.step(batch[1][0], batch[1][1])
            loss_normalisation_constant = batch[1][0].numel()
            loss=loss/loss_normalisation_constant #Normalise the loss value by the number of elements in the training batch, gives a rough idea of convergence
            losses.append(loss)
        
        if epoch % (num_epochs/5) == 0:
            print(f"Epoch {epoch}:, Loss: {loss}")
            
    plt.scatter(np.linspace(1, len(losses), len(losses)), np.log(np.array(losses)), marker='x')
    plt.title("Loss graph for SVI")
    plt.show()
    
    return guide, losses

def NormalNeuralNetworkForRedshifts():
    #####NORMAL NEURAL NETWORK FOR REDSHIFTS - FOR COMPARISON PURPOSES#####
    class DeterministicNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.Layer1 = nn.Linear(input_dim, hidden_dim)
            self.Layer2 = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, input_data):
            hidden_data = nn.functional.tanh(self.Layer1(input_data))
            output_data = self.Layer2(hidden_data)
            
            return output_data
        
    DNN = DeterministicNN(input_nodes_redshifts, 32, output_nodes_redshifts)
    loss_fn = nn.MSELoss()  # regression
    optimizer = torch.optim.Adam(DNN.parameters(), lr=1e-6)

    n_epochs = 3000
    DNN.train()
    loss_values = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        train_data_tensor_redshifts_reduced = UncertaintySampling(train_data_tensor_redshifts, False, redshifts=True)
        for x in range(train_data_tensor_redshifts_reduced.shape[0]):
            optimizer.zero_grad()
            y_pred = DNN(train_data_tensor_redshifts_reduced).squeeze()
            #print(y_pred, train_redshifts_tensor)
            loss = loss_fn(y_pred, train_redshifts_tensor)
            loss.backward()
            optimizer.step()
            loss_values[epoch] = loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
    plt.scatter(np.linspace(1, loss_values.shape[0], loss_values.shape[0]), loss_values, marker='x', color='orange')
    plt.title("Loss function over epochs")
    plt.show()
    
    DNN.eval()
    train_y_preds = torch.zeros((train_data_tensor_redshifts_reduced.shape[0]))
    for i in range(train_data_tensor_redshifts_reduced.shape[0]):
        train_y_preds[i] = DNN(train_data_tensor_redshifts_reduced[i])
    plt.scatter(train_redshifts_tensor.detach().numpy(), train_y_preds.detach().numpy(), marker='x')
    plt.title("32-node deterministic NN train dataset predictions")
    plt.plot([0,3], [0,3], label='Perfect prediction accuracy', color='black', ls='--')
    plt.legend()
    plt.show()
    
    test_data_tensor_redshifts_reduced = UncertaintySampling(test_data_tensor_redshifts, False, redshifts=True)
    test_y_preds = torch.zeros((test_data_tensor_redshifts_reduced.shape[0]))
    for i in range(test_data_tensor_redshifts_reduced.shape[0]):
        test_y_preds[i] = DNN(test_data_tensor_redshifts_reduced[i])
    plt.scatter(test_redshifts_tensor.detach().numpy(), test_y_preds.detach().numpy(), marker='x')
    plt.title("32-node deterministic NN test dataset predictions")
    plt.plot([0,3], [0,3], label='Perfect prediction accuracy', color='black', ls='--')
    plt.legend()
    plt.show()

'''
redshifts_without_class_samples_chain_split = {}
for x in redshifts_with_class_samples.keys():
    temp = torch.split(redshifts_with_class_samples[x], 2000)
    print(temp[0].shape)
    temp_2 = torch.stack((temp[1], temp[3], temp[5], temp[7]))
    print(temp_2.shape)
    redshifts_without_class_samples_chain_split[x] = temp_2
    print(redshifts_with_class_samples_no_warmup[x].shape)
'''

###CAN RUN THESE BEFORE ANY MCMC STUFF TO INVESTIGATE INITIAL PREDICTIVE BEHAVIOUR OF CLASSIFICATION NETWORK###
###ADD A RETURN TO THE PYRO.DETERMINISTIC OF THE PROBABILITIES BEFORE RUNNING THESE, SO PROBS ARE RETURNED###
#test = ClassificationModelFunc(train_data_tensor)
#plt.hist(test.detach().numpy())

if __name__=="__main__":
    
    #plt.hist(copy_master_data_array[copy_master_data_array['CLASS'].lower() == 'bll']['Redshift'], color='blue', label='BLLs', bins=30, alpha=0.6)
    #plt.hist(copy_master_data_array[copy_master_data_array['CLASS'].lower() == 'fsrq']['Redshift'], color='red', label='FSRQs', bins=30, alpha=0.6)
    #plt.hist(copy_master_data_array[copy_master_data_array['CLASS'].lower() == 'bcu']['Redshift'], color='green', label='BCUs', bins=30, alpha=0.6)
    #plt.title("Redshift distribution in 4LAC by AGN class")
    #plt.ylabel("Number of objects")
    #plt.xlabel("Redshift")
    #plt.legend()
    
    using_SVI = True
    using_HMC = False
    
    if using_SVI and using_HMC:
        sys.exit("Don't try and use both SVI and HMC at the same time!")
    
    '''
    ############################################# CLASSIFICATION TRAINING #############################################
    if using_HMC:
        list_mcmcs_class = []
        HMC_Model_Class = ClassificationModelFunc
        num_samples_class = 100 
        num_warmup_class = 500 #Per cross_validation run, we have cross_validation_k*(num_samples_class+num_warmup_class) samples done in total
        num_chains_class = 1
        
    if using_SVI:
        losses = []
        if sampled_uncertainties:
            #Using poutine.block prevents SVI from attempting to infer "true" values for each object's feature
            #Since we apply to test data (where we don't know the "true" values anyway), it prevents unnecessary computation and avoid potential biases in the model - effectively we just treat the inputs as noisy!
            SVI_Model_Class = pyro.poutine.block(ClassificationModelFunc, hide=["Flux1000", "Energy_Flux100", "PL_Index", "LP_Index", "LP_beta", "Frac_Variability"])
        else:
            SVI_Model_Class = ClassificationModelFunc    
        
    cross_validation_k_class = 1 #Number of folds to make/number of cross-validation runs to perform. Setting it to 1 just trains one model on the whole training set

    if cross_validation_k_class > 1:
        #CV is useful to show that the model generalises well to unseen data; (hopefully) provides evidence that the precise split in the dataset is unimportant
        skf = sklearn.model_selection.StratifiedKFold(cross_validation_k_class) #Stratified K-fold CV keeps class compositions equal between fold splits
        
        master_accuracies = np.zeros(cross_validation_k_class)
        master_f1s = np.zeros(cross_validation_k_class)
        master_briers = np.zeros(cross_validation_k_class)
        master_aucs = np.zeros(cross_validation_k_class)
        master_lpds = np.zeros(cross_validation_k_class)
        
        auc_total = 0
        fig_roc = plt.figure()
        ax_roc = fig_roc.gca()
        ax_roc.plot([0, 1], [0, 1], ls='--', color='gray', alpha=0.7, label='Random guesses')
        ax_roc.set_xlabel("False Positive Rate (FPR)")
        ax_roc.set_ylabel("True Positive Rate (TPR)")
        ax_roc.set_title(f"ROC curve for model; average AUC score = {auc_total/cross_validation_k_class:.3f}")

        
        for fold, (train_data, test_data) in enumerate(skf.split(train_data_tensor, train_class_tensor)):
            print(f"\nFold {fold+1}")
            
            temp_val_data = train_data_tensor[test_data]
            temp_val_classes = train_class_tensor[test_data]
            
            temp_train_data = train_data_tensor[train_data]
            temp_train_classes = train_class_tensor[train_data]
            
            #SVI TRAINING METHOD FOR CLASSIFICATION
            if using_SVI:
                temp_samples=None
                overflow_1, overflow_2, temp_train_dataloader = InitialiseDataLoaders(temp_train_data, temp_train_classes)
                guide, losses = SVITrainingLoop(temp_train_dataloader, SVI_Model_Class)                
            
            #HMC TRAINING METHOD FOR CLASSIFICATION
            if using_HMC:
                guide=None
                mcmc = MCMCMethod(temp_train_data, temp_train_classes, num_samples_class, num_warmup_class, HMC_Model_Class, num_chains=num_chains_class)
                list_mcmcs_class.append(mcmc)
                if sampled_uncertainties:
                    temp_samples = dict(list(list_mcmcs_class[0].get_samples().items())[5:9])
                else:
                    temp_samples = dict(list(list_mcmcs_class[0].get_samples().items()))
                
            preds, probs, accuracy_rates, f1_scores, brier_scores, auc_scores, lpds = ClassMCMCPlotting(temp_val_data, temp_val_classes, samples=temp_samples, guide=guide, return_metrics=True, plots=True)
                                    
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(temp_val_classes, np.mean(probs, axis=0)[:, 1])
            auc_total += sklearn.metrics.roc_auc_score(temp_val_classes, np.mean(probs, axis=0)[:, 1])
            
            if fold != cross_validation_k_class - 1:
                ax_roc.plot(fpr, tpr, color='purple', alpha=0.3)
            
            else:
                ax_roc.plot(fpr, tpr, label='5-fold CV ROC Curves for 32 node classifier', color='purple', alpha=0.3)
                ax_roc.legend()    
                ax_roc.show()

            master_accuracies[fold] = np.mean(accuracy_rates)
            master_f1s[fold] = np.mean(f1_scores)
            master_briers[fold] = np.mean(brier_scores)
            master_aucs[fold] = np.mean(auc_scores)
            master_lpds[fold] = np.mean(lpds)
        
        print("Average accuracy from each fold: ", master_accuracies, "\nAverage accuracy over all folds: ", np.mean(master_accuracies))
        print("Average F1-score from each fold: ", master_f1s, "\nAverage F1-score over all folds: ", np.mean(master_f1s))
        print("Average Brier score from each fold: ", master_briers, "\nAverage Brier score over all folds: ", np.mean(master_briers))
        print("Average AUC from each fold: ", master_aucs, "\nAverage AUC over all folds: ", np.mean(master_aucs))
        print("Average NLPD from each fold: ", -1*master_lpds, "\nAverage NLPD over all folds: ", -1*np.mean(master_lpds))
        
        print("\nTraining on whole dataset now!")
           
    #Training a single model on all data at once gives the 'most optimal' sample set
    if using_SVI:
        all_samples_class=None
        guide, losses = SVITrainingLoop(train_dataloader, SVI_Model_Class)    

    if using_HMC:
        guide=None
        mcmc = MCMCMethod(train_data_tensor, train_class_tensor, num_samples_class, num_warmup_class, ClassificationModelFunc, num_chains=4)
        list_mcmcs_class.append(mcmc)
        if sampled_uncertainties:
            all_samples_class = dict(list(mcmc.get_samples().items())[5:9])
        else:
            all_samples_class = dict(list(mcmc.get_samples().items()))
    
    
    ClassMCMCPlotting(test_data_tensor, test_class_tensor, samples=all_samples_class, guide=guide, return_metrics=False)
    
    #all_samples_class = ListMergingAndTrimmingMCMCs(list_mcmcs_class[0:5], is_redshifts=False)
    #SaveSamples(all_samples_class, file_name='temp_samples_class_dict.npy')
    
    if using_HMC:
        axes = az.plot_trace(mcmc.get_samples(group_by_chain=True)['Layer2.weight'])
        axes[0][0].set_title("Posterior dist. for layer 2 weights")
        axes[0][0].set_xlabel("Value")
        axes[0][0].set_ylabel("Density")
        axes[0][1].set_title("Trace plots for layer 2 weights")
        axes[0][1].set_xlabel("Sample number")
        axes[0][1].set_ylabel("Value")            
        plt.show()   
    '''
    
    all_samples_class = LoadSamples('standardised_samples_class_dict.npy')
    #all_samples_class = LoadSamples('CLASS_STANDARD_32_nodes_without_input_unc_modelled.npy')
    print("Class training done!")

    

    ############################################# REDSHIFT TRAINING #############################################
    train_data_tensor_redshifts, train_redshifts_tensor, train_source_name_redshifts = DataFormattingForRedshifts(train_data_tensor, train_class_tensor, train_redshift_array, train_source_name_array, all_samples_class, include_classifications=including_classifications_in_training)
    test_data_tensor_redshifts, test_redshifts_tensor, test_source_name_redshifts = DataFormattingForRedshifts(test_data_tensor, test_class_tensor, test_redshift_array, test_source_name_array, all_samples_class, include_classifications=including_classifications_in_training)
    
    #BCU data reduction
    #bcu_temp_data_tensor, bcu_class_mean_probs, bcu_redshift_array, bcu_source_name_array = ClassifyingBCUs(master_data_array, sample_set=all_samples_class, guide=guide)
    #bcu_class_mean_probs_tensor = torch.tensor(np.mean(bcu_class_mean_probs[:, :, 1], axis=0))
    #bcu_data_tensor = torch.cat((bcu_temp_data_tensor, bcu_class_mean_probs_tensor.unsqueeze(1)), dim=1)
    
    #known_bcu_redshift_indices = np.where(bcu_redshift_array != -np.inf)[0]
    #known_bcu_data_tensor, known_bcu_class_mean_probs, known_bcu_redshifts, known_bcu_source_name_array = bcu_data_tensor[known_bcu_redshift_indices], bcu_class_mean_probs_tensor[known_bcu_redshift_indices], bcu_redshift_array[known_bcu_redshift_indices], bcu_source_name_array[known_bcu_redshift_indices]
    #known_bcu_redshifts_tensor = torch.tensor(known_bcu_redshifts)
    
    #Z-score the redshifts to standardise scale - may improve training stability!
    redshifts_z_scored = True
    if redshifts_z_scored:
        redshifts_mean, redshifts_std = torch.mean(train_redshifts_tensor), torch.std(train_redshifts_tensor)
        train_redshifts_tensor = (train_redshifts_tensor-redshifts_mean)/redshifts_std
        test_redshifts_tensor = (test_redshifts_tensor-redshifts_mean)/redshifts_std
        #known_bcu_redshifts_tensor = (known_bcu_redshifts_tensor-redshifts_mean)/redshifts_std
        redshifts_z_score_tuple = (redshifts_mean, redshifts_std)
        low_redshift_indices = torch.where((test_redshifts_tensor*redshifts_z_score_tuple[1])+redshifts_z_score_tuple[0] < 1.5)
    else:
        redshifts_z_score_tuple = ()
        low_redshift_indices = torch.where(test_redshifts_tensor < 1.5)
    
    if using_HMC:
        list_mcmcs_redshift = []
        HMC_Model_Redshift = RedshiftsModelFunc
        num_samples_redshift = 2000
        num_warmup_redshift = 8000
        num_chains_redshift = 1
    
    if using_SVI:
        losses = []
        if sampled_uncertainties:
            #Using poutine.block prevents SVI from attempting to infer "true" values for each object's feature
            #Since we apply to test data (where we don't know the "true" values anyway), it prevents unnecessary computation and avoid potential biases in the model - effectively we just treat the inputs as noisy!
            SVI_Model_Redshift = pyro.poutine.block(RedshiftsModelFunc, hide=["Flux1000", "Energy_Flux100", "PL_Index", "LP_Index", "LP_beta", "Frac_Variability"])
        else:
            SVI_Model_Redshift = RedshiftsModelFunc    
    
        overflow_1, overflow_2, train_dataloader_redshifts = InitialiseDataLoaders(train_data_tensor_redshifts, train_redshifts_tensor)
    
    
    cross_validation_k_redshifts = 1
    
    if cross_validation_k_redshifts > 1:
        #CV is useful to show that the model generalises well to unseen data; (hopefully) provides evidence that the precise split in the dataset is unimportant
        print(f"Starting {cross_validation_k_redshifts}-fold cross-validation now!")
        skf = sklearn.model_selection.KFold(cross_validation_k_redshifts)
        
        master_lpds = np.zeros(cross_validation_k_redshifts)
        master_RMSE = np.zeros(cross_validation_k_redshifts)
        master_r_squared = np.zeros(cross_validation_k_redshifts)
        master_chi_squared_reduced = np.zeros(cross_validation_k_redshifts)
        master_avg_pred_std = np.zeros(cross_validation_k_redshifts)
                
        for fold, (train_data, test_data) in enumerate(skf.split(train_data_tensor_redshifts, train_redshifts_tensor)):
            print(f"\nFold {fold+1}")
            
            temp_val_data = train_data_tensor_redshifts[test_data]
            temp_val_redshifts = train_redshifts_tensor[test_data]
            
            temp_train_data = train_data_tensor_redshifts[train_data]
            temp_train_redshifts = train_redshifts_tensor[train_data]
            
            #SVI TRAINING METHOD FOR REDSHIFTS
            if using_SVI:
                temp_samples=None
                overflow_1, overflow_2, temp_train_dataloader_redshifts = InitialiseDataLoaders(temp_train_data, temp_train_redshifts)
                guide, losses = SVITrainingLoop(temp_train_dataloader_redshifts, SVI_Model_Redshift)                
                
            #HMC TRAINING METHOD FOR REDSHIFTS
            if using_HMC:
                guide=None
                mcmc = MCMCMethod(train_data_tensor_redshifts, train_redshifts_tensor, num_samples_redshift, num_warmup_redshift, HMC_Model_Redshift, num_chains=num_chains_redshift)
                list_mcmcs_redshift.append(mcmc)
                if sampled_uncertainties:
                    temp_samples = dict(list(mcmc.get_samples().items())[5:9])
                else:
                    temp_samples = dict(list(mcmc.get_samples().items()))
                if 'log_sigma' not in temp_samples.keys():
                    temp_samples['log_sigma'] = (torch.zeros_like(temp_samples['Layer2.bias'])+global_redshift_obs_noise_scale).squeeze()
            
            preds, outputs, master_lpds[fold], master_RMSE[fold], master_r_squared[fold], master_chi_squared_reduced[fold] = RedshiftPerformance(temp_val_data, temp_val_redshifts, samples=temp_samples, guide=guide, redshifts_z_scored=redshifts_z_score_tuple, return_metrics=True)
            master_avg_pred_std[fold] = np.std(np.mean(preds, axis=0))
            
        print("Average NLPD from each fold: ", -master_lpds, "\nAverage NLPD over all folds: ", -np.mean(master_lpds))
        print("Average RMSE from each fold: ", master_RMSE, "\nAverage RMSE over all folds: ", np.mean(master_RMSE))
        print("Average R^2 score from each fold: ", master_r_squared, "\nAverage R^2 score over all folds: ", np.mean(master_r_squared))
        print("Average chi-squared from each fold: ", master_chi_squared_reduced, "\nAverage chi-squared over all folds: ", np.mean(master_chi_squared_reduced))
        print("Average std. from each fold: ", master_avg_pred_std, "\nAverage std. over all folds: ", np.mean(master_avg_pred_std))
    
        print("\nK-fold CV finished, training on all data now!")
        
    if using_SVI:
        all_samples_redshift=None
        guide, losses = SVITrainingLoop(train_dataloader_redshifts, SVI_Model_Redshift)
           
        predictive = pyro.infer.Predictive(SVI_Model_Redshift, guide=guide, num_samples=200)
        samples=predictive(train_data_tensor_redshifts)
        print("Mean log_sigma:", samples["log_sigma"].mean(0))

    if using_HMC:
        guide=None
        mcmc = MCMCMethod(train_data_tensor_redshifts, train_redshifts_tensor, num_samples_redshift, num_warmup_redshift, HMC_Model_Redshift, num_chains=4)
        list_mcmcs_redshift.append(mcmc)
        
        if sampled_uncertainties:
            all_samples_redshift = dict(list(mcmc.get_samples().items())[5:9])
        else:
            all_samples_redshift = dict(list(mcmc.get_samples().items()))
        if not global_redshift_noise_sampling:
            all_samples_redshift['log_sigma'] = (torch.zeros_like(all_samples_redshift['Layer2.bias'])+global_redshift_obs_noise_scale).squeeze()
            
        #all_samples_redshift = ListMergingAndTrimmingMCMCs(list_mcmcs_redshift, is_redshifts=True)
        #SaveSamples(all_samples_redshift, file_name='temp_samples_redshift_dict_full_mass_4_chains_no_input_unc.npy')
    
    #all_samples_redshift = LoadSamples('temp_samples_redshift_dict_diag_mass_4_chains.npy')
        
    #Training data predictions
    print("\nTraining Dataset Fitting:")
    RedshiftPerformance(train_data_tensor_redshifts, train_redshifts_tensor, samples=all_samples_redshift, guide=guide, redshifts_z_scored=redshifts_z_score_tuple, plots=True)
    
    #Test data predictions
    print("\nTest Dataset Fitting:")
    RedshiftPerformance(test_data_tensor_redshifts, test_redshifts_tensor, samples=all_samples_redshift, guide=guide, redshifts_z_scored=redshifts_z_score_tuple, plots=True)
    
    #Low-redshift test data only!
    print("\nLow-Redshift Test Dataset Fitting:")
    RedshiftPerformance(test_data_tensor_redshifts[low_redshift_indices], test_redshifts_tensor[low_redshift_indices], samples=all_samples_redshift, guide=guide, redshifts_z_scored=redshifts_z_score_tuple, plots=True)

    #BCU Predictions
    #print("\nBCU Dataset Fitting:")
    #RedshiftPerformance(known_bcu_data_tensor, known_bcu_redshifts_tensor, samples=all_samples_redshift, redshifts_z_scored=(redshifts_mean, redshifts_std), plots=True)

    #Layer 2 weights sample plot (HMC only!)
    if using_HMC:
        axes = az.plot_trace(mcmc.get_samples(group_by_chain=True)['Layer2.weight'])
        axes[0][0].set_title("Posterior dist. for layer 2 weights")
        axes[0][0].set_xlabel("Value")
        axes[0][0].set_ylabel("Density")
        axes[0][1].set_title("Trace plots for layer 2 weights")
        axes[0][1].set_xlabel("Sample number")
        axes[0][1].set_ylabel("Value")
        plt.show()
        
        #print("CLASSIFIER Layer 2 weights ESS:", pyro.ops.stats.effective_sample_size(list_mcmcs_class[0].get_samples()['Layer2.weight'].unsqueeze(0)))
        #print("CLASSIFIER Layer 2 weights split-Rhat:", pyro.ops.stats.split_gelman_rubin(list_mcmcs_class[0].get_samples()['Layer2.weight'].unsqueeze(0)))
        print("REDSHIFTS Layer 2 weights ESS:", pyro.ops.stats.effective_sample_size(mcmc.get_samples(group_by_chain=True)['Layer2.weight']))
        print("REDSHIFTS Layer 2 weights split-Rhat:", pyro.ops.stats.split_gelman_rubin(mcmc.get_samples(group_by_chain=True)['Layer2.weight']))
        try:
            print("Mean and std. of log-sigma scale after training:", mcmc.get_samples()['log_sigma'].mean(), "+/-", mcmc.get_samples()['log_sigma'].std())
        except:
            print("Log_sigma wasn't sampled!")
    
    
    finish = time.time()
    print("Run time:", finish-start, "seconds")
    print("Used feature values sampled based on uncertainties: ", sampled_uncertainties)
    print(f"Used {hidden_nodes_classification} hidden nodes for classifier and a log prior width of {prior_scale_class}.\nUsed {hidden_nodes_redshifts} hidden nodes for redshift predictor and a log prior width of {prior_scale_redshift}, and an observation noise scale of {global_redshift_obs_noise_scale}.")
        
    #loaded_samples = LoadSamples('temp_samples_dict.npy')
            
    #time.sleep(10)
    #os.system("shutdown.exe /h")
    
    '''
    for i in range (1, 6):
        temp_samples = {key: value[:, :(i*1000)] for key, value in mcmc.get_samples(group_by_chain=True).items()}
        temp_samples = {key: value.flatten(end_dim=1) for key, value in temp_samples.items()}
        print(temp_samples['Layer2.weight'].shape)
        temp_samples['log_sigma'] = (torch.zeros_like(temp_samples['Layer2.bias'].squeeze())-10)
        RedshiftPerformance(test_data_tensor_redshifts, test_redshifts_tensor, samples=temp_samples, redshifts_log_transformed=redshifts_log_transformed, plots=True)
    '''  
    
def OuterUncSampling(input_features, input_classes, samples):
    num_samples_marginalising = 500
    master_probs = np.zeros((num_samples_marginalising, samples['Layer2.weight'].shape[0], input_features.shape[0], 2))
    
    non_unc_inputs = [0, 1, 2, 7, 10, 15, 16, 17, 18, 19, 22]
    
    fsrqs = np.where(input_classes.cpu().numpy() == 1)
    blls = np.where(input_classes.cpu().numpy() == 0)
    
    lpd_total = 0
    
    for i in range(num_samples_marginalising):
        Flux1000 = torch.distributions.Normal(input_features[:, 3], input_features[:, 4]).sample()
        Energy_Flux100 = torch.distributions.Normal(input_features[:, 5], input_features[:, 6]).sample()
        PL_Index = torch.distributions.Normal(input_features[:, 8], input_features[:, 9]).sample()
        LP_Index = torch.distributions.Normal(input_features[:, 11], input_features[:, 12]).sample()
        LP_beta = torch.distributions.Normal(input_features[:, 13], input_features[:, 14]).sample()
        Frac_Variability = torch.distributions.Normal(input_features[:, 20], input_features[:, 21]).sample()
        
        temp = torch.stack((Flux1000, Energy_Flux100, PL_Index, LP_Index, LP_beta, Frac_Variability))
        temp = torch.transpose(temp, 0, 1)
        
        input_features_with_samples = torch.index_select(input_features, 1, torch.LongTensor(non_unc_inputs))
        input_features_with_samples = torch.cat((input_features_with_samples, temp), dim=1)
        
        if using_HMC:
            predictive = pyro.infer.Predictive(model=HMC_Model_Redshift, posterior_samples=samples, return_sites={"obs_class", "probabilities"})
        if using_SVI:
            predictive = pyro.infer.Predictive(model=SVI_Model_Redshift, guide=guide, num_samples=500, return_sites={"obs_class", "probabilities"})
        
        output = predictive(input_features_with_samples)
        preds, probs = output['obs_class'].cpu().numpy(), output['probabilities'].squeeze().cpu().numpy()
        
        master_probs[i] = probs
        
        mean_probs = np.mean(probs, axis=0)
        fsrq_log_probs = np.log(mean_probs[fsrqs, 1]).squeeze()
        bll_log_probs = np.log(mean_probs[blls, 0]).squeeze()
        
        #print(fsrq_log_probs.shape)
        
        lpd = np.sum(fsrq_log_probs, axis=0) + np.sum(bll_log_probs, axis=0)
        #print(f"Run {i+1} Mean LPD:", lpd/mean_probs.shape[0])
        
        lpd_total += lpd/mean_probs.shape[0]
        
        if (i+1) % 20 == 0:
            print(f"{i+1} steps, mean LPD after marginalising over input unc.:", lpd_total/i)

    print("Mean LPD after marginalising over input unc.:", lpd_total/num_samples_marginalising)
    
'''
fig = plt.figure()
ax = fig.gca()
sns.kdeplot(list_mcmcs_class[0].get_samples()['Layer2.weight'][:, 0, 1], ax=ax)
sns.kdeplot(list_mcmcs_class[0].get_samples()['Layer2.weight'][:, 0, 8], ax=ax)
sns.kdeplot(list_mcmcs_class[0].get_samples()['Layer2.weight'][:, 0, 12], ax=ax)
ax.set_xlabel("Weight Value")
ax.set_ylabel("Probability Density")
ax.set_title("Posterior weight distribution for three different weights")
plt.show()
'''

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
file_to_load = 'Classifier Node Number Selection\\NEW_individual_32_nodes.npy'
imported_samples = LoadSamples(file_to_load)
split_samples = DictSplit(imported_samples)

cross_validation_k_class = 5
skf = sklearn.model_selection.StratifiedKFold(cross_validation_k_class) #Stratified K-fold CV keeps class compositions equal between fold splits

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