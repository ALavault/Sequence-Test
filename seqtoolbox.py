#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:33:17 2017

@author: viper

Description : Toolbox for sequences of images
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

def gradientDescent(gradient,nIter, markers,useRandom=False):
    """
    Naive 8 directions gradient descent
    Inputs :
        - gradient : matrix representing the gradient to descend
        - nIter : Maximum number of iterations to descend
        - markers : List of coordinates representing the original seeds
    Output :
        markers : a seed list of coordinates according to the descent
    """
    nbIter = 0

    if useRandom:
        # Deprecated
        raise NameError('HiThere')
    else:
        w, h = gradient.shape
        nMarkers = len(markers) 
        for i in range(nIter):
            nbIter+=1
            count=0
            for k in range(nMarkers):
                x,y = int(markers[k][0]), int(markers[k][1])
                grad = gradient[x,y]
                neighboursList=[(x,max(0,y-1)), (max(x-1,0),y), (x,min(h-1,y+1)), (min(x+1,w-1),y), (max(x-1,0),max(0,y-1)), (max(x-1,0),min(h-1,y+1)), (min(x+1,w-1),min(h-1,y+1)), (min(x+1,w-1),max(0,y-1))] # L, T, R, B, TL, TR, BR, BL
                distance = [gradient[a,b]-grad for a,b in neighboursList]
                l = np.argmin(distance) # Get the index of the minimum delta
                d = np.min(distance) # Get the value of the minimum delta
                if d<=0: # If a point is lower, update the markers
                    a,b = neighboursList[l]
                    markers[k] = [a,b]
                else:
                    count+=1
            if count==nMarkers: # If all neighbours are not interesting
                    break
        return markers, nbIter

def resizeMarkers(markers, ratio):
    newMarkers = [[int(i*ratio), int(j*ratio)] for i,j in markers]
    return np.asarray(newMarkers)

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

def labelExtractor(labels, markers):
    l=[labels[int(x),int(y)] for x,y in markers]
    w,h = labels.shape
    for i in range(w):
        for j in range(h):
            if not labels[i,j] in l:
                labels[i,j] = 0
    return labels
                
