import os
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import time
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.io import imread, imsave

########################################################################33
def alignImages(image0, image1):

    image0 = rgb2gray(image0)
    image1 = rgb2gray(image1)

    orb = ORB(n_keypoints=500, fast_threshold=0.05)

    print("   detect im0")
    orb.detect_and_extract(image0)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    print("   detect im1")
    orb.detect_and_extract(image1)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Select keypoints from the source (image to be registered)
    # and target (reference image)
    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]

    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=4, residual_threshold=1, max_trials=300)
    print("Matched points found = {}".format(inliers.size))
    return (model_robust)



###################################################################################

if __name__ == "__main__":

    image_files = sorted(os.listdir("input"))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)

    images = []
    for img in image_files:
        print ("Reading in file {}".format(img))
        images.append(imread("input/{}".format(img)))

    n = 0
    print("Image  {}".format(n))

    imsave("aligned/aligned{:02d}.jpg".format(n), images[n])

    for n in range (1, len(images) ):
        print("Image align {}".format(n))
        model_robust = alignImages(images[n-1], images[n])
        print("    warping")
        images[n] = warp(images[n], model_robust.inverse)
        images[n] = np.uint8(images[n]*255.0)
        imsave("aligned/aligned{:02d}.jpg".format(n), images[n])
        print("    Image save {}".format(n))

    print ("That's All Folks!")