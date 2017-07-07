#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:24:31 2017
@author: viper
Description : Test of segmentation of a sequence of images
"""
from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import exposure
from skimage import transform

from skimage.feature import (corner_harris, corner_subpix, corner_peaks,plot_matches)
from skimage import morphology
import matplotlib
from skimage.feature import ORB, match_descriptors
matplotlib.use('pgf')
import numpy as np
from skimage import measure

from skimage import color
from skimage import feature
from skimage import filters
from skimage import segmentation
from skimage.measure import ransac
from skimage import util
import time as t
import regiongrowing as rg
import seqtoolbox as classifier
import time
plt.close('all')

nfolder = 2
nbIter= 25

fileList = os.listdir(os.getcwd()+'/move'+str(nfolder)+'-angle')
print(len(fileList),len(fileList)//4)
fileList.sort()
toTreat = []
# Init
image0 = io.imread('move'+str(nfolder)+'-angle/'+fileList[0])
#image0 = image0[:224,:]
plt.imshow(image0, cmap='gray')

markers = plt.ginput(n=2) # Get the two original seeds
markers=np.asarray(markers) # Python list to Numpy array conversion
x, y = markers.T
plt.imshow(image0, cmap='gray')
    
for i in range(len(markers)): # Convert the seeds in order to be used with Numpy arrays
    x_,y_ = markers[i]
    markers[i]=[y_,x_]
markers.astype(int) # Integer coordinates
plt.close('all')
fnames=['AoP_','DoP_' , 'I0_', 'I45_', 'I90_', 'I135_','S0_', 'S1_', 'S2_']
markersOrigin = markers.copy()
timeList=[]
square = morphology.square(3)
for fname in fnames:
    if fname[0]=='A' or fname[0]=='D':
        folder = 'polar'
    elif fname[0]=='I':
        folder = 'angle'
    elif fname[0] == 'S':
        folder = 'stokes'
    else:
        ()
    markers=markersOrigin.copy()
    for i in range(len(fileList)//4-1):
        print(fname , i)

        # Get a different threshold knowing the kind of file (if region growing algorithm is usedS)
        if fname[0]=='A':
            regT = 6
            pixT= regT
        else:
            regT = 6*2**8
            pixT= regT

        plt.clf() # Clear the figure
        dt = time.time() # Launch the time measurement
        # Open image
        image0 = io.imread('move'+str(nfolder)+'-'+folder+'/'+fname+str(i)+'.tiff')
        image0 = exposure.rescale_intensity(image0) # Contrast enhancement
        hog=filters.scharr(image0)
        hog = filters.sobel(hog) # Edge detector of the edge detctor
        hog = morphology.dilation(hog, square) # Dilation of the edge-edge detector
        hog = filters.gaussian(hog, sigma = 1.5) # Filtering
        hog = exposure.rescale_intensity(hog)
        markers2 = markers.copy()
        markers2, _ = classifier.gradientDescent(hog,markers2,nIter = nbIter,useRandom=False)
        # Getting the segmented region
        #labels = rg.regionGrowing(image0, markers, pixT, regT,hasMaxPoints = True, maxPoints =500) # Region growing based on mearkers
        #labels=classifier.labelExtractor(segmentation.felzenszwalb(image0, scale=1.8, sigma=10, min_size=55, multichannel=False), markers)
        #classifier.getConvexLabels(labels) # Convex hull of the labels
        # Narkers Tracking
        #markers2 = classifier.gradientTracking(util.img_as_float(image0), markers, nbIter= nbIter)
        markers2=np.asarray(markers2)
        timeList.append(time.time() - dt) # Stop the time measurement
        markers2.astype('int16')# Python list to Numpy array conversion
        markers=np.asarray(markers)
        #Plotting
        plt.imshow(image0, cmap = 'gray')
        #plt.imshow(transform.rescale(color.label2rgb(labels, image0), 1), cmap = 'gray')
        plt.axis('off')
        x,y = markers.T
        a, b = markers2.T
        plt.plot(y,x, '+r', ms=6)
        plt.plot(b,a, '+g', ms=6)
        plt.savefig('misc'+str(nfolder)+'/'+fname+str(i)+'.tiff')
        markers= markers2

plt.clf()
hist, bins, _ = plt.hist(timeList, bins = 200)
plt.plot(bins[:-1], hist)
plt.axvline(np.mean(timeList), color='b', linestyle='dashed', linewidth=2)
plt.axvline(np.median(timeList), color='r', linestyle='dashed', linewidth=2)

print(np.mean(timeList))