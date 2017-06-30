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

def gradientDescent(gradient,nIter, markers):
    """
    Naive 8 directions gradient descent
    Inputs :
        - gradient : matrix representing the gradient to descend
        - nIter : Maximum number of iterations to descend
        - markers : List of coordinates representing the original seeds
    Output :
        markers : a seed list of coordinates according to the descent
    """
    w, h = gradient.shape
    nMarkers = len(markers) 
    for i in range(nIter):
        k=0
        for k in range(nMarkers):
            x,y = int(markers[k][0]), int(markers[k][1])
            grad = gradient[x,y]
            neighboursList=[(x,max(0,y-1)), (max(x-1,0),y), (x,min(h-1,y+1)), (min(x+1,w-1),y), (max(x-1,0),max(0,y-1)), (max(x-1,0),min(h-1,y+1)), (min(x+1,w-1),min(h-1,y+1)), (min(x+1,w-1),max(0,y-1))] # L, T, R, B, TL, TR, BR, BL
            distance = []
            for a,b in neighboursList: # get all the gradient deltas
                distance.append(gradient[a,b]-grad)
            l = np.argmin(distance) # Get the index of the minimum delta
            d = np.min(distance) # Get the value of the minimum delta
            if d<=0: # If a point is lower, update the markers
                a,b = neighboursList[l]
                markers[k] = [a,b]
            else:
                k+=1
        if k==nMarkers: # If all neighbours are not interesting
                break
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
