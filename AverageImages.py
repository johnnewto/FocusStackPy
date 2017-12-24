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
from PIL import Image


########################################################################33



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
    image_files = sorted(os.listdir("c:/temp/average"))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)


    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open("c:/temp/average/{}".format(image_files[0])).size
    N = len(image_files)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), np.float)
    for im in image_files:
        imarr = np.array(Image.open("c:/temp/average/{}".format(im)), dtype=np.float32)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save("c:/temp/Average.png")
    out.show()
    print ("That's All Folks!")


if False:
    import os, numpy, PIL
    from PIL import Image

    # Access all PNG files in directory
    allfiles = os.listdir(os.getcwd())
    imlist = [filename for filename in allfiles if filename[-4:] in [".png", ".PNG"]]

    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(imlist[0]).size
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = numpy.zeros((h, w, 3), numpy.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr = numpy.array(Image.open("c:/temp/average/{}".format(im)), dtype=numpy.float)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save("Average.png")
    out.show()