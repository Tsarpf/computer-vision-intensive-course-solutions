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

p = cv2.imread('pattern.pbm', cv2.IMREAD_GRAYSCALE) / 255
m = cv2.imread('messi-gray.tiff', cv2.IMREAD_GRAYSCALE) / 255

p_ga1 = clip(p + random_gauss(p, 0.05))
p_ga2 = clip(p + random_gauss(p, 0.15))
# Tehtävä 5
gaussed_m1 = clip(m + random_gauss(m, 0.05))
gaussed_m2 = clip(m + random_gauss(m, 0.15))

# Tehtävä 6
def random_saltpepper(A, r):
    B = A + np.random.binomial(1, r/2, A.shape) # pepper
    C = B * np.random.binomial(1, 1 - (r/2), A.shape) # salt
    return C

# Tehtävä 7
#TODO clip
pattern_peppered1 = clip(random_saltpepper(p, 0.1)) / 255
pattern_peppered2 = clip(random_saltpepper(p, 0.5)) / 255
m_peppered1 = clip(random_saltpepper(m, 0.1)) / 255
m_peppered2 = clip(random_saltpepper(m, 0.5)) / 255
#cv2.imshow('joku iqqnan nimi',  random_saltpepper(gaussed_messi1, 0.2))
#cv2.waitKey()

# Tehtävä 8
def lowpass(A, n):
    weights = [[1 for x in range(n)] for y in range(n)]
    return nd.filters.convolve(A, weights) / (n*n)

# Tehtävä 9

pattern_lp1 = lowpass(p, 3)
pattern_lp2 = lowpass(p, 5)
gaussed_pattern1_lp1 = lowpass(p_ga1, 3)
gaussed_pattern1_lp2 = lowpass(p_ga1, 5)
gaussed_pattern2_lp1 = lowpass(p_ga2, 3)
gaussed_pattern2_lp2 = lowpass(p_ga2, 5)
pattern_peppered1_lp1 = lowpass(pattern_peppered1, 3) 
pattern_peppered1_lp2 = lowpass(pattern_peppered1, 5) 
pattern_peppered2_lp1 = lowpass(pattern_peppered2, 3) 
pattern_peppered2_lp2 = lowpass(pattern_peppered2, 5) 


m_lp1 = lowpass(m, 3)
m_lp2 = lowpass(m, 5)
gaussed_m1_lp1 = lowpass(gaussed_m1, 3)
gaussed_m1_lp2 = lowpass(gaussed_m1, 5)
gaussed_m2_lp1 = lowpass(gaussed_m2, 3)
gaussed_m2_lp2 = lowpass(gaussed_m2, 5)
m_peppered1_lp1 = lowpass(m_peppered1, 3) 
m_peppered1_lp2 = lowpass(m_peppered1, 5) 
m_peppered2_lp1 = lowpass(m_peppered2, 3) 
m_peppered2_lp2 = lowpass(m_peppered2, 5) 

# Tehtävä 10
def highpass(A, n):
    w = 25
    weights = [[(-1) for x in range(n)] for y in range(n)]
    x = int(n/2)
    y = int(n/2)
    weights[x][y] = w 
    return nd.filters.convolve(A, weights) / (n*n)


### Tehtävä 11
# TODO: clippaus re: tehtävänanto
pattern_hp1 = highpass(p, 3)
pattern_hp2 = highpass(p, 5)
gaussed_pattern1_hp1 = highpass(p_ga1, 3)
gaussed_pattern1_hp2 = highpass(p_ga1, 5)
gaussed_pattern2_hp1 = highpass(p_ga2, 3)
gaussed_pattern2_hp2 = highpass(p_ga2, 5)
pattern_peppered1_hp1 = highpass(pattern_peppered1, 3) 
pattern_peppered1_hp2 = highpass(pattern_peppered1, 5) 
pattern_peppered2_hp1 = highpass(pattern_peppered2, 3) 
pattern_peppered2_hp2 = highpass(pattern_peppered2, 5) 

m_hp1 = highpass(m, 3)
m_hp2 = highpass(m, 5)
gaussed_m1_hp1 = highpass(gaussed_m1, 3)
gaussed_m1_hp2 = highpass(gaussed_m1, 5)
gaussed_m2_hp1 = highpass(gaussed_m2, 3)
gaussed_m2_hp2 = highpass(gaussed_m2, 5)
m_peppered1_hp1 = highpass(m_peppered1, 3) 
m_peppered1_hp2 = highpass(m_peppered1, 5) 
m_peppered2_hp1 = highpass(m_peppered2, 3) 
m_peppered2_hp2 = highpass(m_peppered2, 5) 

### Tehtävä 12
def median(A, n):
    return nd.filters.median_filter(A, n)

### Tehtävä 13
pattern_md1 = median(p, 3)
pattern_md2 = median(p, 5)
gaussed_pattern1_md1 = median(p_ga1, 3)
gaussed_pattern1_md2 = median(p_ga1, 5)
gaussed_pattern2_md1 = median(p_ga2, 3)
gaussed_pattern2_md2 = median(p_ga2, 5)
pattern_peppered1_md1 = median(pattern_peppered1, 3) 
pattern_peppered1_md2 = median(pattern_peppered1, 5) 
pattern_peppered2_md1 = median(pattern_peppered2, 3) 
pattern_peppered2_md2 = median(pattern_peppered2, 5) 

m_md1 = median(m, 3)
m_md2 = median(m, 5)
gaussed_m1_md1 = median(gaussed_m1, 3)
gaussed_m1_md2 = median(gaussed_m1, 5)
gaussed_m2_md1 = median(gaussed_m2, 3)
gaussed_m2_md2 = median(gaussed_m2, 5)
m_peppered1_md1 = median(m_peppered1, 3) 
m_peppered1_md2 = median(m_peppered1, 5) 
m_peppered2_md1 = median(m_peppered2, 3) 
m_peppered2_md2 = median(m_peppered2, 5) 

### Tehtävä 14
cv2.imwrite('p.jpg', cv2.resize(p * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga1.jpg', cv2.resize(p_ga1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga2.jpg', cv2.resize(p_ga2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp1.jpg', cv2.resize(pattern_peppered1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp2.jpg', cv2.resize(pattern_peppered2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))

cv2.imwrite('p-l3.jpg', cv2.resize(pattern_lp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-l5.jpg', cv2.resize(pattern_lp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga1-l3.jpg', cv2.resize(gaussed_pattern1_lp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga1-l5.jpg', cv2.resize(gaussed_pattern1_lp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga2-l3.jpg', cv2.resize(gaussed_pattern2_lp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga2-l5.jpg', cv2.resize(gaussed_pattern2_lp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp1-l3.jpg', cv2.resize(pattern_peppered1_lp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp1-l5.jpg', cv2.resize(pattern_peppered1_lp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp2-l3.jpg', cv2.resize(pattern_peppered2_lp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp2-l5.jpg', cv2.resize(pattern_peppered2_lp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))

cv2.imwrite('p-h3.jpg', cv2.resize(pattern_hp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-h5.jpg', cv2.resize(pattern_hp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga1-h3.jpg', cv2.resize(gaussed_pattern1_hp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga1-h5.jpg', cv2.resize(gaussed_pattern1_hp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga2-h3.jpg', cv2.resize(gaussed_pattern2_hp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga2-h5.jpg', cv2.resize(gaussed_pattern2_hp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp1-h3.jpg', cv2.resize(pattern_peppered1_hp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp1-h5.jpg', cv2.resize(pattern_peppered1_hp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp2-h3.jpg', cv2.resize(pattern_peppered2_hp1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp2-h5.jpg', cv2.resize(pattern_peppered2_hp2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))

cv2.imwrite('p-m3.jpg', cv2.resize(pattern_md1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-m5.jpg', cv2.resize(pattern_md2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga1-m3.jpg', cv2.resize(gaussed_pattern1_md1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga1-m5.jpg', cv2.resize(gaussed_pattern1_md2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga2-m3.jpg', cv2.resize(gaussed_pattern2_md1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-ga2-m5.jpg', cv2.resize(gaussed_pattern2_md2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp1-m3.jpg', cv2.resize(pattern_peppered1_md1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp1-m5.jpg', cv2.resize(pattern_peppered1_md2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp2-m3.jpg', cv2.resize(pattern_peppered2_md1 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('p-sp2-m5.jpg', cv2.resize(pattern_peppered2_md2 * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))


cv2.imwrite('m.jpg', cv2.resize(m * 255, None, fx=20, fy=20, interpolation=cv2.INTER_AREA))
cv2.imwrite('m-ga1.jpg', gaussed_m1 * 255)
cv2.imwrite('m-ga2.jpg', gaussed_m2 * 255)
cv2.imwrite('m-sp1.jpg', m_peppered1 * 255)
cv2.imwrite('m-sp2.jpg', m_peppered2 * 255)

cv2.imwrite('m-l3.jpg', m_lp1 * 255)
cv2.imwrite('m-l5.jpg', m_lp2 * 255)
cv2.imwrite('m-ga1-l3.jpg', gaussed_m1_lp1 * 255)
cv2.imwrite('m-ga1-l5.jpg', gaussed_m1_lp2 * 255)
cv2.imwrite('m-ga2-l3.jpg', gaussed_m2_lp1 * 255)
cv2.imwrite('m-ga2-l5.jpg', gaussed_m2_lp2 * 255)
cv2.imwrite('m-sp1-l3.jpg', m_peppered1_lp1 * 255)
cv2.imwrite('m-sp1-l5.jpg', m_peppered1_lp2 * 255)
cv2.imwrite('m-sp2-l3.jpg', m_peppered2_lp1 * 255)
cv2.imwrite('m-sp2-l5.jpg', m_peppered2_lp2 * 255)

cv2.imwrite('m-h3.jpg', m_hp1 * 255)
cv2.imwrite('m-h5.jpg', m_hp2 * 255)
cv2.imwrite('m-ga1-h3.jpg', gaussed_m1_hp1 * 255)
cv2.imwrite('m-ga1-h5.jpg', gaussed_m1_hp2 * 255)
cv2.imwrite('m-ga2-h3.jpg', gaussed_m2_hp1 * 255)
cv2.imwrite('m-ga2-h5.jpg', gaussed_m2_hp2 * 255)
cv2.imwrite('m-sp1-h3.jpg', m_peppered1_hp1 * 255)
cv2.imwrite('m-sp1-h5.jpg', m_peppered1_hp2 * 255)
cv2.imwrite('m-sp2-h3.jpg', m_peppered2_hp1 * 255)
cv2.imwrite('m-sp2-h5.jpg', m_peppered2_hp2 * 255)

cv2.imwrite('m-m3.jpg', m_md1 * 255)
cv2.imwrite('m-m5.jpg', m_md2 * 255)
cv2.imwrite('m-ga1-m3.jpg', gaussed_m1_md1 * 255)
cv2.imwrite('m-ga1-m5.jpg', gaussed_m1_md2 * 255)
cv2.imwrite('m-ga2-m3.jpg', gaussed_m2_md1 * 255)
cv2.imwrite('m-ga2-m5.jpg', gaussed_m2_md2 * 255)
cv2.imwrite('m-sp1-m3.jpg', m_peppered1_md1 * 255)
cv2.imwrite('m-sp1-m5.jpg', m_peppered1_md2 * 255)
cv2.imwrite('m-sp2-m3.jpg', m_peppered2_md1 * 255)
cv2.imwrite('m-sp2-m5.jpg', m_peppered2_md2 * 255)

### Tehtävä 15
### Tehtävä 16
m_uint8 = (m * 255).astype(np.uint8)
m_hist = cv2.equalizeHist(m_uint8)

h = [0]*255
for i in m_uint8.flatten():
    h[i] += 1
    
#plt.plot(h)
#plt.savefig('messi_uint8_hist.png')
#plt.show()

h2 = [0]*256
for i in m_hist.flatten():
    h2[i] += 1

plt.plot(h2)
plt.savefig('messi_hist.png')
#plt.show()
#### Tehtävä 16

#cv2.imshow('joku iqqnan nimi',  cv2.resize(median(messi, 2), None, fx=2, fy=2, interpolation=cv2.INTER_AREA))
#cv2.imshow('joku iqqnan nimi',  cv2.resize(messi_hist, None, fx=2, fy=2, interpolation=cv2.INTER_AREA))
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