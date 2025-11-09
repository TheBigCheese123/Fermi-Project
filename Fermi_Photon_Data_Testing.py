# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:26:58 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import astropy
from astropy.io import fits

fits_file = fits.open("lat_photon_weekly_w908_p305_v001.fits")

fits_file[0].header

data = fits_file[1].data
pls_work = data[data.field("energy") > 1000]

number_of_points = 20000

energy = pls_work.field("ENERGY")[0:number_of_points]
l = pls_work.field("l")[0:number_of_points]
b = pls_work.field("b")[0:number_of_points]

i=0
for i in range(len(l)):
    if l[i] > 180:
        l[i] = (l[i]-360)#*(2*np.pi)/360
    else:
        l[i] = l[i]#*(2*np.pi)/360
        
b = b*(2*np.pi)/360

test_cmap = matplotlib.cm.plasma

energy=np.log10(energy)
norm = matplotlib.colors.Normalize(vmin=np.min(energy), vmax=np.max(energy))
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=test_cmap)

plt.style.use('dark_background')
plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='aitoff')
ax.grid(True)
ax.scatter(l, b, color=test_cmap(norm(energy)), s=4)
cbar = fig.colorbar(mapper, ax=ax)
cbar.ax.set_title("$log_{10}(E)$")
ax.set_title("Detected photons of $E > 10^4$ MeV by Fermi-LAT (23-30 Oct 2025)", y=1.1)
#ax.scatter(l-180, b, color='pink', s=5)

plt.show()
