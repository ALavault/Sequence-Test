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
from skimage import data
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,plot_matches)
from skimage import morphology
import matplotlib
from skimage.feature import ORB, match_descriptors
matplotlib.use('pgf')
import numpy as np
from skimage import color
from skimage import feature
from skimage import filters
from skimage import segmentation
from skimage.measure import ransac
from skimage import util
import time as t
import regiongrowing as rg
import seqtoolbox as classifier



fileList = os.listdir(os.getcwd()+'/move1-angle')
fileList.sort()
toTreat = []
# Init
image0 = io.imread('move1-angle/'+fileList[0])
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
for fname in fnames:
    print(fname)
    if fname[0]=='A' or fname[0]=='D':
        folder = 'polar'
    elif fname[0]=='I':
        folder = 'angle'
    elif fname[0] == 'S':
        folder = 'stokes'
    else:
        ()
    markers=markersOrigin.copy()
    for i in range(212):
        # Get a different threshold knowing the kind of file
        if fname =='AoP_':
            regT = 500//64
            pixT= regT
        else:
            regT = 570
            pixT= regT

        print(i)
        plt.clf() # Clear the figure
        # Open images
        image0 = io.imread('move1-'+folder+'/'+fname+str(i)+'.tiff')
        image1 = io.imread('move1-'+folder+'/'+fname+str(i+1)+'.tiff')
        # "Tracking treatment
        hog= filters.scharr(image0) # Edge detector
        hog2 = filters.sobel(hog) # Edge detector of the edge detctor
        hog = (hog2)/np.max(hog2) # Normalization
        hog = morphology.dilation(hog, morphology.square(3)) # Dilation of the edge-edge detector
        hog = filters.gaussian(hog, sigma = 0.5) # Filtering
        hog = color.rgb2gray(hog) # Convert to grayscale (optionnal)
        markers2 = markers.copy()
        markers2 = classifier.gradientDescent(hog, 2, markers2) # Gradient Descent -> Give new markers
        #markers2 = classifier.resizeMarkers(markers2, 1/ratio)
        markers2=np.asarray(markers2) # Python list to Numpy array conversion
        markers=np.asarray(markers)
        a, b = markers2.T
        labels = rg.regionGrowing(image0, markers, pixT, regT,hasMaxPoints = True) # Region growing based on mearkers
        
        #model_robust, inliers, outliers, src, dst = fpm.featurePointMatching(image0, image1, decimation = 1)
        #markers = fpm.getNewMarkers(model_robust, markers)
        #plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, hog), labels))
        
        #Plotting
        plt.imshow(hog, cmap='gray')   
        plt.axis('off')
        x,y = markers.T
        a, b = markers2.T
        plt.plot(y,x, '+r', ms=6)
        plt.plot(b,a, '+g', ms=6)
        markers= markers2
        plt.savefig('misc/'+fname+str(i)+'.tiff')
        

    
    