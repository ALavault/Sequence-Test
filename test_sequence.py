#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:24:31 2017

@author: viper

"""
from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import data
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,plot_matches)
from skimage import transform
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
import featurepoint_matching as fpm
import classification_test as classifier
from skimage import feature

fileList = os.listdir(os.getcwd()+'/move1-angle')
fileList.sort()
toTreat = []
# Init
image0 = io.imread('move1-angle/'+fileList[0])
plt.imshow(image0, cmap='gray')

markers = plt.ginput(n=2)
markers=np.asarray(markers) # Python list to Numpy array conversion

x, y = markers.T
plt.imshow(image0, cmap='gray')
    
for i in range(len(markers)):
    x_,y_ = markers[i]
    markers[i]=[y_,x_]
markers.astype(int)
plt.close('all')
pixT = 2000
regT = 450
fnames=['S0_', 'S1_', 'S2_','AoP_','DoP_', 'I0_', 'I45_', 'I90_', 'I135_' ]
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
    for i in range(20):
        plt.clf()
        image0 = io.imread('move1-'+folder+'/'+fname+str(i)+'.tiff')
        #image1 = io.imread('move1-polar/I0_'+str(i+1)+'.tiff')
        # HOG
        #fd, hog = feature.hog(image0, orientations=16, pixels_per_cell=(5, 5), cells_per_block=(1, 1), visualise=True)
        hog= filters.scharr(image0) # Edge detector
        hog = filters.gaussian(hog) # Filtering
        markers2 = markers.copy()
        markers2 = classifier.gradientDescent(hog, 2, markers2) # Descent
        #markers2 = classifier.resizeMarkers(markers2, 1/ratio)
        markers2=np.asarray(markers2) # Python list to Numpy array conversion
        markers=np.asarray(markers)
        a, b = markers2.T
        labels = rg.regionGrowing(image0, markers, pixT, regT)
        #model_robust, inliers, outliers, src, dst = fpm.featurePointMatching(image0, image1, decimation = 1)
        #markers = fpm.getNewMarkers(model_robust, markers)
        plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, image0), labels))
        #plt.imshow(image0, cmap='gray')   
        x,y = markers.T
        a, b = markers2.T
        plt.plot(y,x, 'or', ms=4)
        plt.plot(b,a, 'og', ms=4)
        markers= markers2
        plt.savefig('misc/'+fname+str(i)+'.tiff')
        

    
    