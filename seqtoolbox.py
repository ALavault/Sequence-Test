#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:33:17 2017

@author: viper

Description : Toolbox for sequences of images
"""
import numpy as np
from skimage import measure
from skimage import filters
from skimage import morphology
from skimage import exposure



def gradientDescent(gradient,markers, nIter=25 ,useRandom=False):
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
    """
    Give resized markers. Should be used with resized images.
    Inputs :
        - markers : List of coordinates representing the original seeds
        - ratio : size ratio to give to the markers
    Output :
        newMarkers : resized markers as Numpy array
    """

    newMarkers = [[int(i*ratio), int(j*ratio)] for i,j in markers]
    return np.asarray(newMarkers)

def labelExtractor(labels, markers):
    """
    Extract labeled area where the markers are.
    Inputs :
        - labels : an image/2D array containing labeled regions
        - markers : seeds to test
    Output :
        - labels : Image/2D array containing only the labels where the markers are.
    """
    l=[labels[int(x),int(y)] for x,y in markers]
    w,h = labels.shape
    for i in range(w):
        for j in range(h):
            if not labels[i,j] in l:
                labels[i,j] = 0
    return labels
                
def getConvexLabels(labels):
    """
    Give the convex hull of given labels. 0 means background and so is not processed.
    """
    props = measure.regionprops(labels)
    bbox = [ props[k].bbox for k in range(len(props))]
    cv = [ props[k].convex_image for k in range(len(props))]
    for k in range(len(props)):
        min_row, min_col, max_row, max_col = bbox[k]
        labels[min_row:max_row, min_col:max_col] = (k+1)*cv[k]
    return labels

def gradientTracking(image, markers, nbIter =25, selem = morphology.square(3), sigma = 1.5):
    hog=filters.scharr(image)
    hog = filters.sobel(hog) # Edge detector of the edge detctor
    hog = morphology.dilation(hog, selem) # Dilation of the edge-edge detector
    hog = filters.gaussian(hog, sigma = sigma) # Filtering
    hog = exposure.rescale_intensity(hog)
    markers2 = markers.copy()
    markers2, _ = gradientDescent(hog,markers,nbIter,useRandom=False)
    return markers2