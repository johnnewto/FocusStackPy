import os
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import time
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import warp, downscale_local_mean, resize, SimilarityTransform
from skimage.io import imread, imsave
import time


########################################################################33

def detectFeatures(img, resiz):
    orb = ORB(n_keypoints=500, fast_threshold=0.05)
    img = rgb2gray(img)
    img = resize(img, (img.shape[0] * resiz, img.shape[1] * resiz))
    orb.detect_and_extract(img)
    return orb.keypoints, orb.descriptors


def matchFeatures(keypoints1, descriptors1, keypoints2, descriptors2):
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Select keypoints from the source (image to be registered) and target (reference image)
    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]
    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=4, residual_threshold=1, max_trials=300)
    return model_robust, inliers


###################################################################################

if __name__ == "__main__":
    # resiz is set so to make feature detect faster
    resiz = 0.25   # reduces from 6 seconds down to 2.3 seconds
    image_files = sorted(os.listdir("input"))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)

    images = []
    for imgN in image_files:
        print ("Reading in file {}".format(imgN))
        img = imread("input/{}".format(imgN))
        images.append(img)

    start = time.time()
    n = 0
    print("Image  {}".format(n))
    imsave("aligned/aligned{:02d}.jpg".format(n), images[n])

    print("   detecting features")
    keypoints1, descriptors1 = detectFeatures(images[n],resiz)
    print("   Time Elapsed = {:.3f}".format(time.time() - start))

    tform = SimilarityTransform(scale=1)

    for n in range (1, len(images) ):
        print("Image align {}".format(n))
        # print("    detecting features")
        keypoints2, descriptors2 = detectFeatures(images[n], resiz)

        tform2, inliers = matchFeatures(keypoints1, descriptors1, keypoints2, descriptors2)
        print("    Matched points found = {}".format(inliers.size))

        # add to the transform and warp the image
        print("    warping scale {:.3f}, traslation {:.3f} {:.3f}".format(tform2.scale, tform2.translation[0], tform2.translation[1]))
        tform2.translation[0] /=resiz
        tform2.translation[1] /=resiz
        tform = tform + tform2
        images[n] = warp(images[n], tform.inverse)

        print("    Image save {}".format(n))
        images[n] = np.uint8(images[n]*255.0)
        imsave("aligned/aligned{:02d}.jpg".format(n), images[n])

        # Keep keypoints for next image
        keypoints1 = keypoints2
        descriptors1 = descriptors2
        print("   Time Elapsed = {:.3f}".format(time.time() - start))



    print ("That's All Folks!")