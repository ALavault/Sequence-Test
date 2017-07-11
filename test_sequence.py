#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:24:31 2017
@author: viper
Description : Test of segmentation of a sequence of images
    This is just a script made to test if the gradient tracking and the segmentation could work together.
    This file also gives after the whole treatment an histogram of the execution time of the treatment (without the marker generation)
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
from skimage import measure
from skimage import color
from skimage import feature
from skimage import filters
from skimage import segmentation
from skimage import util
import time
import regiongrowing as rg
import seqtoolbox as sqtlbx
plt.close('all')



nfolder = 3
nbIter= 20

fileList = os.listdir(os.getcwd()+'/move'+str(nfolder)+'-angle')
fileList.sort()

# Init
image0 = io.imread('move'+str(nfolder)+'-angle/'+fileList[0])
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

# A few constants
fnames=['AoP_','DoP_' , 'I0_', 'I45_', 'I90_', 'I135_','S0_', 'S1_', 'S2_']
markersOrigin = markers.copy()
markerList=[markers]
timeList=[]
square = morphology.square(3)

# Markers generation using AoP -> the most robust of all images for this and simulate a kind of parallel treatment.
print('Markers generation')

for i in range(len(fileList)//4-1):
    image0 = io.imread('move'+str(nfolder)+'-polar/AoP_'+str(i)+'.tiff')
    markers2 = sqtlbx.gradientTracking(image0, markerList[i], nbIter= nbIter)
    markerList.append(markers2)
    
print('Markers generated')
os.nice(1)
# Main treatment
print('Processing...')
for fname in fnames:
    print(fname)
    if fname[0]=='A' or fname[0]=='D':
        folder = 'polar'
    elif fname[0]=='I':
        folder = 'angle'
    elif fname[0] == 'S':
        folder = 'stokes'
    for i in range(len(fileList)//4-1):
        markers = markerList[i]
        markers2 = markerList[i+1]
        #print(fname , i)
        ### Get a different threshold knowing the kind of file (if region growing algorithm is used) ###
        if fname[0]=='A':
            regT = 6
            pixT= regT
        else:
            regT = 6*2**8
            pixT= regT
        plt.clf() # Clear the figure
        ### Open image ###
        dt = time.time() # Launch the time measurement
        image0 = io.imread('move'+str(nfolder)+'-'+folder+'/'+fname+str(i)+'.tiff')
        #image0 = exposure.rescale_intensity(image0) # Contrast enhancement (use the full dynamic according to the type of the image)
        ### Getting the segmented region ###
        labels = rg.regionGrowing(image0, markers, pixT, regT,hasMaxPoints = True, maxPoints =500) # Region growing based on markers
        sqtlbx.getConvexLabels(labels) # Convex hull of the labels, fuller regions
        timeList.append(time.time() - dt) # Stop the time measurement and append the result for further evalutation
        """
        ### Plotting ###
        plt.imshow(color.label2rgb(labels, image0), cmap = 'gray')
        plt.axis('off')
        x,y = markers.T
        a, b = markers2.T
        plt.plot(y,x, '+r', ms=6)
        plt.plot(b,a, '+g', ms=6)
        plt.savefig('misc'+str(nfolder)+'/'+fname+str(i)+'.tiff')
        markers= markers2
        """
### Histogram of execution times
print('Processing Done')
plt.clf()
hist, bins, _ = plt.hist(timeList, bins = 200)
mean = plt.axvline(np.mean(timeList), color='b', linestyle='dashed', linewidth=2, label = 'Mean')
median =plt.axvline(np.median(timeList), color='r', linestyle='dashed', linewidth=2, label = 'Median')
plt.axvline(1/30., color='b', linewidth=1)
plt.axvline(0.1, color='b', linewidth=1)
plt.text(1/30., np.max(hist), '30 fps')
plt.text(0.10, np.max(hist), '10 fps')
plt.xlabel('Time (s)')
plt.axvspan(np.mean(timeList)-np.std(timeList)/2, np.mean(timeList)+np.std(timeList)/2, color='black', alpha=0.25)
plt.legend([mean, median], ['Mean', 'Median'])
print('Average execution time : ' + str(np.mean(timeList)))