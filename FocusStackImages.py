import os
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.color import rgb2gray, gray2rgb
import time
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import resize, rescale
from skimage.io import imread, imsave
from skimage.morphology import reconstruction

import time


########################################################################33

def focusmeasureLAPD(img, filtersiz):
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

def focusmeasureHELM(Image, filtersiz):
    from scipy.ndimage import convolve
    from scipy.ndimage import correlate1d
    from scipy.ndimage.filters import uniform_filter
    # case 'HELM' %Helmli's mean method (Helmli2001)
    #     U = imfilter(Image, MEANF, 'replicate');
    #     R1 = U./Image;
    #     R1(Image==0)=1;
    #     index = (U>Image);
    #     FM = 1./R1;
    #     FM(index) = R1(index);
    #     FM = imfilter(FM, MEANF, 'replicate');
    #     end
    np.seterr(divide='ignore')
    Image = rgb2gray(Image)
    U = uniform_filter(Image, size=filtersiz, mode='reflect')

    with np.errstate(divide='ignore', invalid='ignore'):
        R1 = np.divide(U, Image)
        R1[R1 == np.inf] = 0
        R1 = np.nan_to_num(R1)

    R1[Image==0] = 1
    index = (U > Image)
    with np.errstate(divide='ignore', invalid='ignore'):
        FM = np.divide(1., R1)
        FM[FM == np.inf] = 0
        FM = np.nan_to_num(FM)

    FM[index] = R1[index]
    FM = uniform_filter(FM, size=filtersiz, mode='reflect')
    return FM


def CalcIndex(images):
    start = time.time()
    shp = images[0].shape

    # if shp[0] > 2000:
    #     fm = np.zeros((int(shp[0]/2), int(shp[1]/2), len(images)))
    # else:
    fm = np.zeros((int(shp[0]), int(shp[1]), len(images)))

    print("   focus measure")
    for n in range (0, len(image_files) ):
        print("    In Image{}".format(n))
        fm[:,:,n] = focusmeasureHELM(images[n], 31)
        print("     fmeasure {}".format(np.mean(fm[n])))

        print("     Time Elapsed = {:.3f}".format(time.time() - start))
        im = np.uint8(gray2rgb(fm[n]) * 255.0)

    index = np.argmax(fm, axis=2)
    index = fill(index)


    heights = np.uint8(index * 255.0 / np.max(index))

    return index, heights

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
    mask = np.uint8(np.zeros((shp[0], shp[1], 3, len(images))))
    stack = np.uint8(np.zeros((shp[0], shp[1], 3)))

    for n in range(0, len(images)):
        m = (np.where([index == n],1,0).reshape(shp[0], shp[1]))
        mask[:,:,0,n ] = m
        mask[:,:,1,n ] = m
        mask[:,:,2,n ] = m
        focusmask = np.multiply(mask[:,:,:,n ], images[n])
        print (" Saving mask {}".format(n))
        imsave("stacked/mask{:02d}.jpg".format(n), focusmask)
        stack = np.add(stack,focusmask)
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
        # if img.shape[0] > 2000:
        #     # img = resize(img, (img.shape[0] / 2, img.shape[1] / 2))
        #     img = rescale(img, 0.5)


        # images[:,:,:,n] =img
        images.append(img)
        n = n + 1


    if True:
        index, heights = CalcIndex(images)
        imsave("stacked/HeightMap.jpg", heights)
        np.save('stacked/index.npy', index)

    index = np.load('stacked/index.npy')
    heights = imread("stacked/HeightMap.jpg")

    start = time.time()
    stack = CalcStack(index, images)
    stack = np.uint8(stack)
    # stack = rescale(stack, 2)
    # stack = np.uint8(stack*255)
    imsave("stacked/stack1.jpg", np.uint8(stack))
    print("   Time Elapsed = {:.3f}".format(time.time() - start))

    fig, (ax0, ax1) = plt.subplots(1, 2,
                                 subplot_kw={'xticks': [], 'yticks': []},
                                 figsize=(12, 8))
    cax = ax0.imshow(heights, cmap=cm.hot)
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    ax1.imshow(stack)
    plt.show()
    print ("That's All Folks!")



