# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:50:39 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import astropy
from astropy.io import fits

#4FGL-DR3 - gll_psc_v31.fit
#4FGL-DR4 - gll_psc_v35.fit
#4LAC-DR3 (high-altitude) - table-4LAC-DR3-h.fits

fits_file = fits.open("gll_psc_v35.fit")
fits_file = fits.open("table-4LAC-DR3-h.fits")


data_table = fits_file[1].data
headers = fits_file[1].header
print(headers)

print(np.unique(data_table['CLASS'], return_counts=True))

fits_file = fits.open("table-4LAC-DR3-h.fits")

data_table = fits_file[1].data

#print(fits_file[1].header)
#print(len(data_table), len(data_table[(data_table['CLASS   '] == 'bll') | (data_table['CLASS   '] == 'fsrq') | (data_table['CLASS   '] == 'bcu')]))

#print(data_table[34:54]['REDSHIFT'])

redshift_table = data_table[data_table['REDSHIFT'] != -np.inf]#[21:26]
#print(np.unique(redshift_table['CLASS'], return_counts=True))

b=redshift_table["GLAT"]
l=redshift_table["GLON"]
print(max(l), max(b))
#print(b, l)


i=0
for i in range(len(l)):
    if l[i] > 180:
        l[i] = (l[i]-360)*(2*np.pi)/360
    else:
        l[i] = l[i]*(2*np.pi)/360
        
b = b*(2*np.pi)/360

AGN_labels_full = np.char.lower(redshift_table['CLASS'])
for i in range(len(AGN_labels_full)):
    x = AGN_labels_full[i]
    if (x != 'bll') & (x != 'bcu') & (x != 'fsrq'):
        AGN_labels_full[i] = 'other'
        
AGN_labels_unique = np.unique(AGN_labels_full)
print(AGN_labels_unique)
#AGN_colour_dict = {'agn':0, 'bcu':1, 'bll':2, 'css':3, 'fsrq':4, 'nlsy1':5, 'rdg':6, 'ssrq':7, 'sey':8}
AGN_colour_dict = {'bcu': 'lime', 'bll': 'lightblue', 'fsrq': 'red', 'other': 'magenta'}

plt.style.use('dark_background')
plt.rcParams["figure.autolayout"] = True
plt.rcParams['axes.titley'] = 1.05    # y is in axes-relative coordinates.
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='mollweide')
ax.grid(True)
scatter = ax.scatter(l, b, c=[AGN_colour_dict[label] for label in AGN_labels_full], s=5)
plt.title("Sources (|b| > 10 degrees) present in the 4LAC-DR3 catalog (red=FSRQ, blue=BLL, lime=BCU, magenta=all others)")
#legend1 = ax.legend(*scatter.legend_elements(), labels = AGN_labels_unique, loc='lower center', bbox_to_anchor=(0.5, -0.1))
#ax.add_artist(legend1)

#cbar = plt.colorbar(mapper, ax=ax, ticks=range(len(AGN_labels_unique)), label='Class')
#cbar.ax.set_yticklabels(AGN_labels_unique)
