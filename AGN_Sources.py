# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 17:52:38 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import astropy
from astropy.io import fits
import torch
import seaborn as sns

fits_file = fits.open("table-4LAC-DR3-h.fits")

hdu_table = fits_file[1]

data_table = hdu_table.data
headers = hdu_table.header
#print(headers)
headers = np.array(headers)

#start = 27
#stop = len(headers)-1


features = headers[[27, 30, 33, 36, 41, 43, 45, 48, 50, 52, 54]]
features = np.append(features, ['TTYPE36', 'TTYPE37', 'TTYPE38', 'TTYPE39', 'TTYPE30']) #Adds nu_syn, nuFnu_synvar_index, frac_var, 4and redshift data
#features = headers[np.linspace(start, stop, (stop-start)+1, dtype=int)]
#for x in features:
#    print(hdu_table.header[x])
    
print(len(data_table))
data_table = data_table[data_table['REDSHIFT'] != -np.inf]
results = data_table['REDSHIFT']

#print(data_table[data_table['CLASS'] == 'bcu']['REDSHIFT'])

test = np.zeros((len(features), len(data_table['REDSHIFT'])))

#print(test.shape)

for i in range(len(features)):
    temp_header = hdu_table.header[features[i]]
    #print(temp_header)
    #print(data_table[temp_header])
    test[i, :] = data_table[temp_header]

#print(features)
#print(data_table[0][features])

pls_work = np.corrcoef(test)
#print(pls_work)

features_names = ["" for i in range(len(features))]
for i in range(len(features)):
    features_names[i] = hdu_table.header[features[i]]
    #print(min(data_table[features_names[i]]), max(data_table[features_names[i]]))

sns.heatmap(pls_work, cmap='berlin')
ax = plt.gca()
ax.invert_xaxis()
ax.set_xticklabels(features_names, rotation=90)
ax.set_yticklabels(features_names, rotation=0)
plt.show()

print(np.std(data_table[data_table['CLASS'] == 'fsrq']['PL_Index']), np.std(data_table[data_table['CLASS'] == 'bll']['PL_Index']))

'''
for num in range(len(features)):
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(test[-1, :], test[num,:])
    ax.set_xlabel(features_names[-1])
    ax.set_ylabel(features_names[num])
    plt.show()
'''