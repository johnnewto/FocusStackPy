import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import warp, downscale_local_mean, resize, SimilarityTransform
from skimage.io import imread, imsave
import time
import threading


########################################################################33

def detectFeatures(img, resiz, keypoints, descriptors):
    orb = ORB(n_keypoints=500, fast_threshold=0.05)
    img = rgb2gray(img)
    img = resize(img, (img.shape[0] * resiz, img.shape[1] * resiz), mode = 'reflect')
    orb.detect_and_extract(img)
    keypoints.append(orb.keypoints)
    descriptors.append(orb.descriptors)

    # return orb.keypoints, orb.descriptors


def matchFeatures(keypoints1, descriptors1, keypoints2, descriptors2):
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Select keypoints from the source (image to be registered) and target (reference image)
    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]
    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=4, residual_threshold=1, max_trials=300)
    return model_robust, inliers


def detectAllFeatures(images, resiz):
    print("   detecting features")
    keypoints = []
    descriptors = []
    for n in range (0, len(images) ):
        detectFeatures(images[n], resiz, keypoints, descriptors)
    print("   Time Elapsed = {:.3f}".format(time.time() - start))
    return keypoints, descriptors

def detectAllFeaturesThreaded(images, resiz):
    print("   detecting features")
    keypoints = []
    descriptors = []
    threads = []
    for n in range (0, len(images) ):
        t = threading.Thread(target=detectFeatures, args=(images[n], resiz, keypoints, descriptors,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("   Time Elapsed = {:.3f}".format(time.time() - start))
    return keypoints, descriptors

def matchAllFeatures(keypoints, descriptors):
    print("   matching features")
    tform = []
    tform.append(SimilarityTransform(scale=1))
    for n in range (1, len(images) ):
        tf, inliers = matchFeatures(keypoints[n-1], descriptors[n-1], keypoints[n], descriptors[n])
        tf.translation[0] /=resiz
        tf.translation[1] /=resiz
        tform.append(tf)
        print("   Time Elapsed = {:.3f}".format(time.time() - start))
    return tform

def warpWrapper(images, n, tf):
    images[n] = warp(images[n], tf.inverse)

def warpFeatures(images, tform):
    print("   warping features")
    tf = tform[0]
    for n in range(1, len(images)):
        tf = tf + tform[n]
        # images[n] = warp(images[n], tf.inverse)
        wrapWarp(images, n, tf)
    print("   Time Elapsed = {:.3f}".format(time.time() - start))

def warpFeaturesThreaded(images, tform):
    print("   warping features")
    threads = []
    tf = tform[0]
    for n in range(1, len(images)):
        tf = tf + tform[n]
        t = threading.Thread(target=warpWrapper, args=(images, n, tf,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    print("   Time Elapsed = {:.3f}".format(time.time() - start))

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

    keypoints, descriptors = detectAllFeatures(images, resiz)
    # keypoints, descriptors = detectAllFeaturesThreaded(images, resiz)

    tform = matchAllFeatures(keypoints, descriptors)

    warpFeaturesThreaded(images, tform)
    # warpFeatures(images, tform)

    for n in range(1, len(images)):
        print("    Image save {}".format(n))
        images[n] = np.uint8(images[n]*255.0)
        imsave("aligned/aligned{:02d}.jpg".format(n), images[n])


    print ("That's All Folks!")