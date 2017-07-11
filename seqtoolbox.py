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



def gradientDescent(gradient,markers, nIter=25):
    """
    Naive 8 directions gradient descent (this is not a gradient descent solution to solve linear systems)
    Inputs :
        - gradient : matrix representing the gradient to descend
        - nIter : Maximum number of iterations to descend
        - markers : List of coordinates representing the original seeds
    Output :
        markers2 : a list of coordinates according to the descent (not in place).
    """
    markers2 = markers.copy()
    w, h = gradient.shape
    nMarkers = len(markers) 
    for i in range(nIter):
        count=0
        for k in range(nMarkers):
            x,y = int(markers2[k][0]), int(markers2[k][1])
            grad = gradient[x,y]
            neighboursList=[(x,max(0,y-1)), (max(x-1,0),y), (x,min(h-1,y+1)), (min(x+1,w-1),y), (max(x-1,0),max(0,y-1)), (max(x-1,0),min(h-1,y+1)), (min(x+1,w-1),min(h-1,y+1)), (min(x+1,w-1),max(0,y-1))] # L, T, R, B, TL, TR, BR, BL
            distance = [gradient[a,b]-grad for a,b in neighboursList]
            l = np.argmin(distance) # Get the index of the minimum delta
            d = np.min(distance) # Get the value of the minimum delta
            if d<=0: # If a point is lower, update the markers
                a,b = neighboursList[l]
                markers2[k] = [a,b]
            else:
                count+=1
        if count==nMarkers: # If all neighbours are not interesting
                break
    return markers2

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
    Extract labeled area where the markers are. Useful when a lot of regions are segmented (e.g Superpixel algorithm).
    The extraction consists in making non marked labels part of the background.
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
    Give the convex hull of given labels. 0 means background and so it is not processed.
    Inputs :
        - labels : an image/2D array containing labeled regions
    Output : 
        - labels : modified labels
    Note : this treatment is done in place
    """
    props = measure.regionprops(labels)
    bbox = [ props[k].bbox for k in range(len(props))]
    cv = [ props[k].convex_image for k in range(len(props))]
    for k in range(len(props)):
        min_row, min_col, max_row, max_col = bbox[k]
        labels[min_row:max_row, min_col:max_col] = (k+1)*cv[k]
    return labels

def gradientTracking(image, markers, nbIter =25, selem = morphology.square(3), sigma = 1.5):
    """
    Generate the new markers given an image. 
    First, it generates a gradient-like image according to the following steps :
        1) Scharr filter
        2) Sobel filter of (1)
    This gets a approximation of laplacian operator.
        3) morphological dilation
        4) gaussian filter
    Note : AoP works best with this.
    Inputs :
        - image : original image.
        - markers :original seeds (corrdinates like matrices i.e. [row, column])
        - nbIter : optionnal, give the maximum number of iterations for the gradient descent
        - selem : optionnal, the ele;ent used for dilation
        - sigma : optionnal,
    Outputs :
        - markers2 : a list of coordinates according to the descent (not in place). These are the new seeds according to the gradient descent.
        
    TODO : make the process more customizable.
    """
    hog=filters.scharr(image)
    hog = filters.sobel(hog) # Edge detector of the edge detctor
    hog = morphology.dilation(hog, selem) # Dilation of the edge-edge detector
    hog = filters.gaussian(hog, sigma = sigma) # Filtering
    markers2 = gradientDescent(hog,markers, nIter = nbIter)
    return markers2

def extractPointsFromLabels(labels, useConvexHull = False):
    """
    Extract 2 points from each label.
    The method chosen is simply taking the point at the top left corner of the  label and the point at the bottom right of the same label.
    If the label is reduced to one point, the same point is added twice.
    TODO : show a warning if such problem occurs
    Inputs :
        - labels : label inage obtained with any segmentation method. 0 is not considered as a valid label (background). Has N distinct labels.
        - useConvexHull : optionnal, use original label or its convex hull depending of the value of this argument.
    Outputs : 
        - pointSet : format (N, 2). Each line represents two points from a same label.
    """
    # Generation of a convex hull of labeled regions if set to True
    props = measure.regionprops(labels)
    if useConvexHull:
        bbox = [ props[k].bbox for k in range(len(props))]
        cv = [ props[k].convex_image for k in range(len(props))]
        for k in range(len(props)):
            min_row, min_col, max_row, max_col = bbox[k]
            labels[min_row:max_row, min_col:max_col] = (k+1)*cv[k]
        props = measure.regionprops(labels)

    # Extract some points (top left and bottom right of the non convex region, 
    # simpler than searching the diameter but doesn't assure a maximized distance between the points)
    pointSet = []
    for k in range(len(props)):
        p1, p2 = props[k].coords[0], props[k].coords[-1]
        pointSet.append([p1, p2])
    return pointSet        

        
