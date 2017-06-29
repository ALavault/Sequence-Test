#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:12:37 2017

@author: viper


"""
# Modules....
import matplotlib
matplotlib.use('pgf')
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage import filters
from skimage import feature

from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from skimage import data
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

def getTransform(image1, image2):
    kp1 = feature.corner_harris(image1)
    kp2 = feature.corner_harris(image2)


""" Auxiliary functions """

coords_orig = corner_peaks(corner_harris(image1), threshold_rel=0.001,
                           min_distance=5)
coords_warped = corner_peaks(corner_harris(image2),
                             threshold_rel=0.001, min_distance=5)