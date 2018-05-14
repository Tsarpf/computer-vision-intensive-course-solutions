#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd


def random_gauss(A, r):
    return np.random.normal(0, r, A.shape)

def clip(i):
    return np.clip(i, 0, 1)

pattern = cv2.imread('pattern.pbm', cv2.IMREAD_GRAYSCALE) / 255
messi = cv2.imread('messi-gray.tiff', cv2.IMREAD_GRAYSCALE) / 255

gaussed_pattern1 = clip(pattern + random_gauss(pattern, 0.05))
gaussed_pattern2 = clip(pattern + random_gauss(pattern, 0.15))
# Tehtävä 5
gaussed_messi1 = clip(messi + random_gauss(messi, 0.05))
gaussed_messi2 = clip(messi + random_gauss(messi, 0.15))

# Tehtävä 6
def random_saltpepper(A, r):
    B = A + np.random.binomial(1, r/2, A.shape) # pepper
    C = B * np.random.binomial(1, 1 - (r/2), A.shape) # salt
    return C

# Tehtävä 7
#TODO clip
pattern_peppered1 = clip(random_saltpepper(pattern, 0.1)) / 255
pattern_peppered2 = clip(random_saltpepper(pattern, 0.5)) / 255
messi_peppered1 = clip(random_saltpepper(messi, 0.1)) / 255
messi_peppered2 = clip(random_saltpepper(messi, 0.5)) / 255

# Tehtävä 8
def lowpass(A, n):
    weights = [[1 for x in range(n)] for y in range(n)]
    return nd.filters.convolve(A, weights) / (n*n)

# Tehtävä 9

pattern_lp1 = lowpass(pattern, 3)
pattern_lp2 = lowpass(pattern, 5)
gaussed_pattern1_lp1 = lowpass(gaussed_pattern1, 3)
gaussed_pattern1_lp2 = lowpass(gaussed_pattern1, 5)
gaussed_pattern2_lp1 = lowpass(gaussed_pattern2, 3)
gaussed_pattern2_lp2 = lowpass(gaussed_pattern2, 5)
pattern_peppered1_lp1 = lowpass(pattern_peppered1, 3) 
pattern_peppered1_lp2 = lowpass(pattern_peppered1, 5) 
pattern_peppered2_lp1 = lowpass(pattern_peppered2, 3) 
pattern_peppered2_lp2 = lowpass(pattern_peppered2, 5) 


messi_lp1 = lowpass(messi, 3)
messi_lp2 = lowpass(messi, 5)
gaussed_messi1_lp1 = lowpass(gaussed_messi1, 3)
gaussed_messi1_lp2 = lowpass(gaussed_messi1, 5)
gaussed_messi2_lp1 = lowpass(gaussed_messi2, 3)
gaussed_messi2_lp2 = lowpass(gaussed_messi2, 5)
messi_peppered1_lp1 = lowpass(messi_peppered1, 3) 
messi_peppered1_lp2 = lowpass(messi_peppered1, 5) 
messi_peppered2_lp1 = lowpass(messi_peppered2, 3) 
messi_peppered2_lp2 = lowpass(messi_peppered2, 5) 

# Tehtävä 10
def highpass(A, n):
    w = 25
    weights = [[(-1) for x in range(n)] for y in range(n)]
    x = int(n/2)
    y = int(n/2)
    weights[x][y] = w 
    print(weights)
    return nd.filters.convolve(A, weights) / (n*n)


### Tehtävä 11
pattern_hp1 = highpass(pattern, 3)
pattern_hp2 = highpass(pattern, 5)
gaussed_pattern1_hp1 = highpass(gaussed_pattern1, 3)
gaussed_pattern1_hp2 = highpass(gaussed_pattern1, 5)
gaussed_pattern2_hp1 = highpass(gaussed_pattern2, 3)
gaussed_pattern2_hp2 = highpass(gaussed_pattern2, 5)
pattern_peppered1_hp1 = highpass(pattern_peppered1, 3) 
pattern_peppered1_hp2 = highpass(pattern_peppered1, 5) 
pattern_peppered2_hp1 = highpass(pattern_peppered2, 3) 
pattern_peppered2_hp2 = highpass(pattern_peppered2, 5) 

messi_hp1 = highpass(messi, 3)
messi_hp2 = highpass(messi, 5)
gaussed_messi1_hp1 = highpass(gaussed_messi1, 3)
gaussed_messi1_hp2 = highpass(gaussed_messi1, 5)
gaussed_messi2_hp1 = highpass(gaussed_messi2, 3)
gaussed_messi2_hp2 = highpass(gaussed_messi2, 5)
messi_peppered1_hp1 = highpass(messi_peppered1, 3) 
messi_peppered1_hp2 = highpass(messi_peppered1, 5) 
messi_peppered2_hp1 = highpass(messi_peppered2, 3) 
messi_peppered2_hp2 = highpass(messi_peppered2, 5) 

### Tehtävä 12
#def median(A, n):
#    weights = [[1 for x in range(n)] for y in range(n)]
#    return nd.filters.convolve(A, weights) / (n*n)

cv2.imshow('joku iqqnan nimi',  cv2.resize(highpass(messi, 5), None, fx=2, fy=2, interpolation=cv2.INTER_AREA))
#cv2.imwrite('gaussed_pattern1.jpg', cv2.resize(gaussed_pattern1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
#cv2.imwrite('gaussed_pattern2.jpg', cv2.resize(gaussed_pattern2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
#cv2.imwrite('derp.jpg', cv2.resize(gaussed_pattern2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))

#cv2.imshow('joku iqqnan nimi',  cv2.resize(gaussed_pattern1, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
#cv2.imshow('joku iqqnan nimi',  gaussed_messi1)
#cv2.imshow('joku iqqnan nimi',  clip(gaussed_messi1 + random_saltpepper(gaussed_messi1, 0.5)))
#cv2.imshow('joku iqqnan nimi',  cv2.resize(random_saltpepper(gaussed_messi1, 0.1), None, fx=2, fy=2, interpolation=cv2.INTER_AREA))
#cv2.imshow('joku iqqnan nimi',  clip(gaussed_messi1 + random_saltpepper(gaussed_messi1, 0.1)))
#cv2.imshow('joku iqqnan nimi',  random_saltpepper(gaussed_messi1, 0.2))
#cv2.imshow('joku iqqnan nimi',  gaussed_messi2)

#cv2.imshow('joku iqqnan nimi',  clip(gaussed_messi1 + random_saltpepper(gaussed_messi1, 0.1)))
#cv2.imshow('joku iqqnan nimi',  cv2.resize(lowpass(pattern, 5), None, fx=20, fy=20, interpolation=cv2.INTER_AREA))

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