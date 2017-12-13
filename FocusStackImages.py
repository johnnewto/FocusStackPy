import os
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb
import time
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import warp, downscale_local_mean, resize, SimilarityTransform
from skimage.io import imread, imsave

import time


########################################################################33

def focusmeasure(img, filtersiz):
    from scipy.ndimage import convolve
    from scipy.ndimage import correlate1d
    from scipy.ndimage.filters import uniform_filter
    # M = [-1 2 - 1];
    # Lx = imfilter(Image, M, 'replicate', 'conv');
    # Ly = imfilter(Image, M', 'replicate', 'conv');
    # FM = abs(Lx) + abs(Ly);
    # FM = mean2(FM);
    img = rgb2gray(img)

    M = np.array([-1, 2, -1])
    img1 = correlate1d(img, M, mode='constant', cval=0.0)
    M = np.transpose(M)
    img2 = correlate1d(img, M, mode='constant', cval=0.0)
    img = np.abs(img1) +  np.abs(img2)

    if filtersiz > 0:
        img = uniform_filter(img, size=filtersiz, mode='reflect')
    return img

def CalcIndex(images):
    start = time.time()
    shp = images[0].shape
    fm = np.zeros((shp[0], shp[1], len(images)))
    print("   detecting features")
    for n in range (0, len(image_files) ):
        print("    In Image{}".format(n))
        fm[:,:,n] = focusmeasure(images[n], 29)
        print("     fmeasure {}".format(np.mean(fm[n])))

        print("     Time Elapsed = {:.3f}".format(time.time() - start))
        im = np.uint8(gray2rgb(fm[n]) * 255.0)

    index = np.argmax(fm, axis=2)
    heights = np.uint8(index * 255 / np.max(index))
    return index, heights

def old_CalcStack(index, images):
    print("   Calc Masks and stacking")
    shp = images[0].shape
    stack = np.uint8(np.zeros((shp[0], shp[1], 3)))
    for n in range(0, np.amax(index)+1):
        m = np.where([index == n],1,0).reshape(shp[0], shp[1])
        a = images[n]
        stack[:,:,0] = np.add(stack[:,:,0],np.multiply(m[:,:], a[:,:,0]))
        stack[:,:,1] = np.add(stack[:,:,1],np.multiply(m[:,:], a[:,:,1]))
        stack[:,:,2] = np.add(stack[:,:,2],np.multiply(m[:,:], a[:,:,2]))
    return stack

def CalcStack(index, images):
    print("   Calc Masks and stacking")
    shp = images[0].shape
    mask = np.uint8(np.zeros((shp[0], shp[1], 3)))
    stack = mask

    for n in range(0, np.amax(index)+1):
        m = (np.where([index == n],1,0).reshape(shp[0], shp[1]))
        mask[:,:,0 ] = m
        mask[:,:,1 ] = m
        mask[:,:,2 ] = m
        stack = np.add(stack,np.multiply(mask, images[n]))
    return stack




###################################################################################

if __name__ == "__main__":

    image_files = sorted(os.listdir("aligned"))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)

    n = 0
    images = []
    for imgN in image_files:
        imgN = image_files[n]
        print ("Reading in file {}".format(imgN))
        img = imread("aligned/{}".format(imgN))
        # img = resize(img, (img.shape[0] / 2, img.shape[1] / 2))
        # images[:,:,:,n] =img
        images.append(img)
        n = n + 1


    if True:
        index, heights = CalcIndex(images)
        imsave("stacked/HeightMap.jpg", heights)
        np.save('stacked/index.npy', index)

    index = np.load('stacked/index.npy')

    start = time.time()
    stack = CalcStack(index, images)
    imsave("stacked/stack1.jpg", np.uint8(stack))
    print("   Time Elapsed = {:.3f}".format(time.time() - start))

    print ("That's All Folks!")



