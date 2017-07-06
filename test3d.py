#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 08:27:55 2017

@author: viper
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from skimage import io
from skimage import filters
from skimage import exposure
from skimage import color
from skimage import morphology
from skimage import feature
from skimage import util
plt.close('all')


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
im = io.imread('AoP_0.tiff')
w,h = im.shape

X = np.arange(0, w, 1)
Y = np.arange(0, h, 1)
w,h = im.shape
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
hog=filters.scharr(im)
hog = filters.sobel(hog)
hog = exposure.rescale_intensity(hog)
 # Edge detector of the edge detctor
hog = exposure.rescale_intensity(hog)
#hog = morphology.dilation(hog, morphology.square(3)) 
#hog = morphology.closing(hog, morphology.square(3)) # Dilation of the edge-edge detector
 
hog = filters.gaussian(hog, sigma = 0.5) # Filtering
hog = exposure.rescale_intensity(hog)
hog = color.rgb2gray(hog) 
th= 0.03
for i in range(w):
    for j in range(h):
        if hog[i,j]>th:
            hog[i,j]=1
        else:
            hog[i,j]=0

        
Z = hog[X,Y]

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=1, antialiased=False)



plt.show()
plt.figure(2)
plt.imshow(hog)