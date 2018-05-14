#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials

import cv2
import numpy as np
import matplotlib.pyplot as plt


def random_gauss(A, r):
    return np.random.normal(0, r, A.shape)

def clip(i):
    return np.clip(i, 0, 1)

def random_saltpepper(A, r):
    print('derp')
    #homma

pattern = cv2.imread('pattern.pbm', cv2.IMREAD_GRAYSCALE) / 255
messi_grey = cv2.imread('messi-gray.tiff', cv2.IMREAD_GRAYSCALE) / 255

gaussed_pattern1 = clip(random_gauss(pattern, 0.05))
gaussed_pattern2 = random_gauss(pattern, 0.15)
# Tehtävä 5
gaussed_messi1 = clip(messi_grey + random_gauss(messi_grey, 0.05))
gaussed_messi2 = clip(messi_grey + random_gauss(messi_grey, 0.15))

# Tehtävä 6
def random_saltpepper(A, r):
    print(A)
    B = A * np.random.binomial(1, 1 - (r/2), A.shape) # pepper
    C = B + np.random.binomial(1, r/2, A.shape) # salt
    return C


cv2.imwrite('derp.jpg', cv2.resize(gaussed_pattern1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('derp.jpg', cv2.resize(gaussed_pattern2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))

#cv2.imshow('joku iqqnan nimi',  cv2.resize(gaussed_pattern1, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
#cv2.imshow('joku iqqnan nimi',  gaussed_messi1)

#cv2.imshow('joku iqqnan nimi',  clip(gaussed_messi1 + random_saltpepper(gaussed_messi1, 0.5)))
cv2.imshow('joku iqqnan nimi',  cv2.resize(random_saltpepper(gaussed_messi1, 0.1), None, fx=2, fy=2, interpolation=cv2.INTER_AREA))
#cv2.imshow('joku iqqnan nimi',  clip(gaussed_messi1 + random_saltpepper(gaussed_messi1, 0.1)))

#cv2.imshow('joku iqqnan nimi',  random_saltpepper(gaussed_messi1, 0.2))

#cv2.imshow('joku iqqnan nimi',  gaussed_messi2)

cv2.waitKey()

#print(cv2.__version__)

#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#print(flags)

#img_bgr = cv2.imread('messi5.jpg')
#print(type(img_bgr), img_bgr.dtype, img_bgr.shape)
#cv2.imshow('BGR', img_bgr)
#cv2.waitKey(0)
#
#img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#print(type(img_g), img_bgr.dtype, img_g.shape)
#cv2.imshow('grey', img_g)
#cv2.waitKey(0)
#
#z = img_bgr
#for i in range(z.shape[0]):
#    for j in range(z.shape[1]):
#        if 1.2*z[i,j,0]>z[i,j,1] or 1.2*z[i,j,2]>z[i,j,1]:
#            z[i,j] = [img_g[i,j]]*3
#
#cv2.imwrite('grass.jpg', z)
#cv2.imshow('grass', z)
#cv2.waitKey(0)
#           
#h = [0]*256
#for i in img_g.flatten():
#    h[i] += 1
#
#plt.plot(h)
#plt.savefig('histogram.png')
#plt.show()
#
#