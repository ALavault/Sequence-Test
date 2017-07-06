#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:48:24 2017

@author: viper
"""

import os
import matplotlib
matplotlib.use('pgf')
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from skimage import color
from skimage import segmentation
from skimage import feature
from skimage import exposure
from  sklearn import cluster
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import PIL.Image as Image

def thresholding(image, mini, maxi):
    image = img_as_float(image)
    labels = np.zeros(image.shape)
    w,h = image.shape
    for i in range(w):
        for j in range(h):
            if image[i,j]>=mini and image[i,j]<=maxi:
                labels[i,j] = 1
            else:
                ()
    return labels

plt.close('all') # Close all remaining figures

fileList = os.listdir(os.getcwd()+'/move1-polar/')
i=1
# Try to get a template from histograms

            

for filename in fileList:
    try :
        if filename[0] == 'D':
            im = io.imread('move1-polar/' + filename)
            im = img_as_ubyte(im)
            hist, bins = exposure.histogram(im)
            m = 1./np.max(bins)
            bins = m*bins
            #plt.plot(bins, hist)
            labels = thresholding(im, 0.2, 0.3)
            plt.imshow(labels)
            plt.savefig('misc2/' + filename)
        
    except IOError :
        print(filename+' : Not a file or not an image (IOError). This file will be skipped.')
