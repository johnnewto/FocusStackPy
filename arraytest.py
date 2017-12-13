import numpy as np

X = np.arange(27).reshape((3, 3, 3))
# print (X)

Y = X.reshape(int(27/ 3) , 3)
# print (Y)

select = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) + np.arange(9)
select = select * 9

a = [4, 3, 5]
b = np.array(a)



import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

image1 = color.rgb2gray(data.astronaut())
image1 = data.astronaut()

image = []
image1 = rescale(image1, 1.0 / 64.0)
image1 = np.array([[[1.0,0.0,0.0], [0.0,0.9,0.0]], [[0.0,0.0,0.8], [0.4,0.4,0.4]]])

image.append(image1)
image.append(image1*0.8)
image.append(image1*0.6)
image.append(image1*0.4)
image = np.array(image)
fig, axes = plt.subplots(nrows=3, ncols=2)

ax = axes.ravel()

# print (image_rescaled)
ax[0].imshow(image[0])
ax[1].imshow(image[1])
ax[2].imshow(image[2])
ax[3].imshow(image[3])


# plt.tight_layout()
# plt.show()

# arrySiz = np.prod(image[0].shape)
# select = np.ones((image[0].shape[0], image[0].shape[1]), dtype=int)
# s1 = np.trunc(np.arange(2).T/2)
# select = np.intp(np.multiply(select, s1))
# select = select * arrySiz;

select = np.array([[0,1], [2,3]])
# step = 3 * np.arange(arrySiz/3).reshape(image[0].shape[0], image[0].shape[1])
# select = select + step
select =np.intp(select)

result = image.choose(select)


# np.take


ax[4].imshow(result)

plt.tight_layout()
plt.show()