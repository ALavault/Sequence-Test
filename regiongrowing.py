#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:19:31 2017
@author: viper
Description : Test of a region growing algorithm
"""

#matplotlib.use('pgf') # Force Matplotlib back-end

# Modules....
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import segmentation
from skimage import color
from skimage import filters
from skimage import img_as_uint

def regionGrowing(image, seeds, pixelThreshold, regionThreshold, labels = None, sortDistances = False, noOrphans = False, hasMaxPoints = False, maxPoints =5000):
    """
    Inputs :
        - image : a grayscale image
        - seeds : an array of seeds in (line, column) coordinates
        - pixelThreshold : threshold between 2 pixels to determine if the tested pixel is valid or not
        - regionThreshold : threhsold between a tested pixel and the seed of the neighbour
        - labels : optionnal, give a preexisting array of labels. Must be the same size as image
        - sortDistances : sort the distance of potential candidates instead of direction, does not seem to be relevant
        - noOrphans : if some pixels are orphans (i.e. their value is 0), bind them to the closest labelled region.
    Output :
        labels : a matrix of segmented regions
    Description : 
        regionGrowing implements a region-growing like algorithm to segment an image
    
    """
    # TODO : Parallelize the process 
    seeds=seeds.astype(int)
    toVisit=seeds.flatten()
    nIter=0
    nSeeds = len(seeds)
    nbRows, nbCols = image.shape
    if labels is None:
        labels = np.zeros(image.shape, dtype=np.int)
    else:
        # TODO : assert that the size of the label matrix is the same as the original image
        ()
    while toVisit != []:
        if hasMaxPoints and nIter>maxPoints:
            break
        #point=(toVisit[0], toVisit[1])
        x,y = int(toVisit[0]), int(toVisit[1])
        seed= labels[x,y]
        nIter+=1
        if nIter<=nSeeds:
            # Initialize the original seeds on matrix labels
            labels[x,y]=nIter
            toVisit = np.append(np.delete(toVisit,[0,1]),(x,y))
        else:
            # Beginning of the treatment
            neighboursList=[(x,max(0,y-1)), (max(x-1,0),y), (x,min(nbCols-1,y+1)), (min(x+1,nbRows-1),y), (max(x-1,0),max(0,y-1)), (max(x-1,0),min(nbCols-1,y+1)), (min(x+1,nbRows-1),min(nbCols-1,y+1)), (min(x+1,nbRows-1),max(0,y-1))] # L, T, R, B, TL, TR, BR, BL
            distances=[]
            # Create a the list of distances in regards of neighboursList
            for candidate in neighboursList:
                a,b = candidate
                distances = np.append(distances, abs(int(image[x,y])-int(image[a,b])))
            toVisitNext=[]
            # Get all possible new points to visit
            for candidate in neighboursList:
                # Test if the point currently processed is acceptable knowing the thresholds and forner labels
                if isAcceptable((toVisit[0], toVisit[1]), candidate, labels, image, seeds, pixelThreshold, regionThreshold): # If it is acceptable,
                    # add it to the list containing the next points to be visited
                    toVisitNext=np.append(toVisitNext, candidate)
                    labels[candidate[0], candidate[1]]= seed
            # Treatment of pixels to be visited        
            if toVisitNext is []: # no acceptable neighbour to visit next
                # then remove the point being treated (the "seed")
                toVisit = np.delete(toVisit,[0,1])
            else:
                # Add the candidates to be visited in the order chosen             
                toVisit = np.append(np.delete(toVisit,[0,1]),toVisitNext)
    # Function to get every non labeled pixel to one of the labeled zones
    if noOrphans:
        # To avoid point with label = 0 pop : array[:-1] 
        stack=[]
        for x in range(nbRows):
            for y in range(nbCols):
                if labels[x,y]==0: #if the point is not labelled
                    neighboursLabels=[labels[x,int(max(0,y-1))], labels[max(x-1,0),y], labels[x,min(nbCols-1,y+1)],labels[min(x+1,nbRows-1),y]]
                    seed = getLabelled(neighboursLabels)
                    if seed is None:
                        stack.append((x,y))
                    else:
                        while stack !=[]:
                            i,j = stack.pop(len(stack)-1)
                            labels[i,j]= seed
                        labels[x,y]=seed
            
    # Return an image containing all the labeled pixelss
    return labels
            
def isAcceptable(point, candidate,labels, image, seeds,pixelThreshold, regionThreshold):
    """
    Inputs :
    Output : boolean
    Description : 
        isAcceptable returns if a candidate from a point is acceptable as a candidate
    """
    x,y = point
    a,b = candidate
    a,b,x,y= int(a),int(b),int(x),int(y)
    seed= labels[x,y] # label of tested point
    originalSeed = seeds[seed-1] # seed of the labeled point
    i,j = originalSeed
    i,j = int(i), int(j)
    return (labels[a,b] ==0 and (abs(int(image[x,y])-int(image[a,b]))<pixelThreshold) and (abs(int(image[i,j])-int(image[a,b]))<regionThreshold))

def getLabelled(neighboursLabels):
    """
    Inputs :
        - neighboursLabels : a list of labels (positive integers)
    Output : value of the most common label in neighboursLabels except zero, None if zero.
    Description : 
        getLabelled returns the most common label in neighboursLabels except zero.
    """
    nbZeros = 0
    toRemove = []
    for k in range(len(neighboursLabels)):
        i = neighboursLabels[k]
        if i==0:
            nbZeros+=1
            toRemove.append(i)
    # If there are only zeros
    if nbZeros == len(neighboursLabels):
        return None
    elif nbZeros >0:
        # Remove all zeros
        for k in toRemove:
            neighboursLabels.remove(k)
        counts = np.bincount(neighboursLabels)

        return np.argmax(counts)
    else:
        counts = np.bincount(neighboursLabels)
        return neighboursLabels[np.argmax(counts)]

def labelExtractor(image):
    nbLabel = np.max(image)
    shape = image.shape
    matrixList=[]
    for i in range(nbLabel):
        matrixList.append(np.zeros(shape, dtype=np.int))
    nbRows, nbCols = shape
    for i in range(nbRows):
        for j in range(nbCols):
            label=image[i,j]
            mat = matrixList[label-1]
            mat[i,j]=label
    return matrixList

"""
Tests......
plt.close('all') # Close all remaining figures
filename = 'S2_0.tiff'
im = io.imread(filename) # Open the image
im = filters.median(im) # filtering : for smoother boundaries
im=img_as_uint(im)
plt.imshow(im, cmap='gray')
markers = plt.ginput(n=3)
markers=np.asarray(markers) # Python list to Numpy array conversion
x, y = markers.T
plt.imshow(im, cmap='gray')
    
for i in range(len(markers)):
    x_,y_ = markers[i]
    markers[i]=[y_,x_]
markers.astype(int)
plt.close('all')
pixT = 2000
regT = 4000
labels = regionGrowing(im, markers, pixT, regT, noOrphans=False)
labels2 = regionGrowing(im, markers, pixT, regT,noOrphans=True)
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))
axarr[0].plot(y, x, 'or', ms=3)
axarr[0].set_title('Without orphan adoption')
axarr[0].axis('off')
axarr[1].imshow(segmentation.mark_boundaries(color.label2rgb(labels2, im), labels2))
axarr[1].plot(y, x, 'or', ms=3)
axarr[1].set_title('With orphan adoption')
axarr[1].axis('off')
plt.savefig('Processed/Region Growing/'+filename, dpi = 96*len(axarr))
"""