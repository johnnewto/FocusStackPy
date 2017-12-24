# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 09:22:02 2017

@author: John
"""
#!/usr/bin/env python

import numpy as np
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

def fill(img):
    seed = np.ones_like(img)*255
    img[ : ,0] = 0
    img[ : ,-1] = 0
    img[ 0 ,:] = 0
    img[ -1 ,:] = 0
    seed[ : ,0] = 0
    seed[ : ,-1] = 0
    seed[ 0 ,:] = 0
    seed[ -1 ,:] = 0
    return reconstruction(seed, img, method='erosion')

# Use the matlab reference Soille, P., Morphological Image Analysis: Principles and Applications, Springer-Verlag, 1999, pp. 208-209.
#  6.3.7  Fillhole
# The holes of a binary image correspond to the set of its regional minima which
# are  not  connected  to  the image  border.  This  definition  holds  for  grey scale
# images.  Hence,  filling  the holes of a  grey scale image comes down  to remove
# all  minima  which  are  not  connected  to  the  image  border, or,  equivalently,
# impose  the  set  of minima  which  are  connected  to  the  image  border.  The
# marker image 1m  used  in  the morphological reconstruction by erosion is set
# to the maximum image value except along its border where the values of the
# original image are kept:

img = imread("tyre.jpg")
f_img = fill(img)
f, (ax0, ax1) = plt.subplots(1, 2,
                                  subplot_kw={'xticks': [], 'yticks': []},
                                  figsize=(12, 8))
ax0.imshow(img)
ax1.imshow(f_img)
plt.show()