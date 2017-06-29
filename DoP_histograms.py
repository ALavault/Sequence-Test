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
plt.close('all') # Close all remaining figures

fileList = os.listdir(os.getcwd()+'/DoPs')
i=1
for filename in fileList:
    try :
        im = io.imread('DoPs/' + filename)
        hist, bins = exposure.histogram(im)
        plt.plot(bins, hist)
    except IOError :
        print(filename+' : Not a file or not an image (IOError). This file will be skipped.')
