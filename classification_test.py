#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:33:17 2017

@author: viper

Description : Test of classification 
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern, hog
from skimage import data
from skimage.color import label2rgb
from skimage import io
from skimage import  color, exposure
from skimage import util
from skimage import transform
from skimage import segmentation
import regiongrowing as rg

def gradientDescent(gradient,nIter, markers):
    w, h = gradient.shape
    nMarkers = len(markers)
    for i in range(nIter):
        k=0
        for k in range(nMarkers):
            x,y = int(markers[k][0]), int(markers[k][1])
            grad = gradient[x,y]
            neighboursList=[(x,max(0,y-1)), (max(x-1,0),y), (x,min(h-1,y+1)), (min(x+1,w-1),y), (max(x-1,0),max(0,y-1)), (max(x-1,0),min(h-1,y+1)), (min(x+1,w-1),min(h-1,y+1)), (min(x+1,w-1),max(0,y-1))] # L, T, R, B, TL, TR, BR, BL
            distance = []
            for a,b in neighboursList:
                distance.append(gradient[a,b]-grad)
            l = np.argmin(distance)
            d = np.min(distance)
            if d<=0:
                a,b = neighboursList[l]
                markers[k] = [a,b]
            else:
                k+=1

    return markers

def resizeMarkers(markers, ratio):
    newMarkers = []
    for k in range(len(markers)):
        i, j = markers[k]
        newMarkers.append([int(i*ratio), int(j*ratio)])
    return newMarkers

def getEstimatedMarkers(markersOrigin, image, nIter=15):
    fd, hogImage = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
    ratio = len(fd)/float(image.size)
    w,h = image.shape
    w, h = w*ratio, h*ratio
    shape = (int(w), int(h))
    fd = np.resize(fd, shape)
    markers2 = resizeMarkers(markersOrigin, ratio)
    markers2 = gradientDescent(fd, nIter, markers2)
    markers2 = resizeMarkers(markers2, 1/ratio)
    return markers2, hogImage
 
"""
 Test of classification 

    
plt.close('all')
image = io.imread('move1-stokes/S1_0.tiff')
image2 = io.imread('move1-stokes/S1_1.tiff')

plt.imshow(image, cmap='gray')
markers = plt.ginput(n=2)
markers=np.asarray(markers) # Python list to Numpy array conversion
x, y = markers.T
   
    
for i in range(len(markers)):
    x_,y_ = markers[i]
    markers[i]=[y_,x_]
markers.astype(int)

fd, hog = hog(image2, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
ratio = len(fd)/float(image.size)
w,h = image.shape
w, h = w*ratio, h*ratio
shape = (int(w), int(h))
fd = np.resize(fd, shape)



markers2 = resizeMarkers(markers, ratio)
markers2 = gradientDescent(fd, 15, markers2)
markers2 = resizeMarkers(markers2, 1/ratio)
markers2=np.asarray(markers2) # Python list to Numpy array conversion
x, y = markers.T
a, b = markers2.T

print(markers, markers2)

toHist = fd
hist, bins = exposure.histogram(toHist)
cdf, bins2 = exposure.cumulative_distribution(toHist)
hist = hist/np.max(hist)
f, axarr = plt.subplots(1, 3)
axarr[0].imshow(image, cmap = 'gray')
axarr[0].plot(y,x, 'or', ms=4)
axarr[0].plot(b,a, 'og', ms=4)

axarr[1].imshow(image2, cmap = 'gray')
axarr[1].plot(y,x, 'or', ms=4)
axarr[1].plot(b,a, 'og', ms=4)

axarr[2].plot(bins, hist)
axarr[2].plot(bins2, cdf)


"""
