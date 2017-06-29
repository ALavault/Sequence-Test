#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:19:31 2017

@author: viper

"""

from __future__ import print_function

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

plt.close('all')
def featurePointMatching(image0, image1, decimation = 1, n_keypoints=750):
    """ As seen at https://peerj.com/articles/453/#fig-5"""
    image0 = transform.rescale(image0,1/float(decimation))
    image1 = transform.rescale(image1, 1/float(decimation))
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.05)
    orb.detect_and_extract(image0)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors
    orb.detect_and_extract(image1)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    # Select keypoints from the source (image to be
    # registered) and target (reference image).
    src = keypoints1[matches12[:, 0]][:, ::-1]
    dst = keypoints2[matches12[:, 1]][:, ::-1]
    model_robust, inliers = \
        ransac((src, dst), transform.EuclideanTransform,
               min_samples=6, residual_threshold=2)
    outliers = inliers == False 
    return model_robust, inliers, outliers, src, dst

def getNewMarkers(model_robust, markers):
    mat = model_robust.params
    markers1 = transform.matrix_transform(markers, mat)
    return markers1

def getRandomMarkersFromLabels(label):
    w, h = label.shape
    marker1 = None
    marker2 = None
    while (marker1 == None) or (marker2 == None):
        i = np.random.randint(w)
        j = np.random.randint(h)
        if label[i,j]==1:
            marker1 = (i, j)
        elif label[i,j]==2:
            marker2 = (i, j)
        else:
            ()
    return np.asarray([marker1, marker2])
"""
 Test 
image0 = io.imread('move1-stokes/S1_0.tiff')
image1 = io.imread('move1-stokes/S1_1.tiff')

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
regT = 1500
labels = rg.regionGrowing(image0, markers, pixT, regT)
model_robust, inliers, outliers, src, dst = featurePointMatching(image0, image1)




# visualize correspondence
print(model_robust.params)
plt.gray()

fig, ax = plt.subplots(nrows=2, ncols=1)

image1 = transform.warp(image1, model_robust)
labels1 = util.img_as_uint(2**(63-16)*transform.warp(labels, model_robust))

ax[0].imshow(labels)
ax[1].imshow(labels1)

f, axarr = plt.subplots(1, 2)


axarr[0].imshow(segmentation.mark_boundaries(color.label2rgb(labels, image0), labels))
axarr[0].plot(y, x, 'or', ms=3)
axarr[0].set_title('')
axarr[0].axis('off')
markers1 = transform.matrix_transform(markers, model_robust.params)
x, y = markers1.T

axarr[1].imshow(segmentation.mark_boundaries(color.label2rgb(labels1, image1), labels1))
axarr[1].plot(y, x, 'or', ms=3)
axarr[1].set_title('')
axarr[1].axis('off')
"""