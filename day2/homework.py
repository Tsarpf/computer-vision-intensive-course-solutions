# aina kun voidaan niin tykitellään.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage as nd
from scipy.spatial import distance

print(cv2.__version__)


# Implement integral image iif(x,y) and apply it to strawberries-binary.pbm after scaling it to a binary image

def summed_area_table(A):
    new = np.full((A.shape[0] + 1, A.shape[1] + 1), 0)
    print(new.shape)
    for x in range(A.shape[0]):
        if x == 0: continue
        for y in range(A.shape[1]):
            if y == 0: continue
            new[x,y] = new[x,y-1] + new[x-1,y] + A[x,y] - new[x-1,y-1]
    return new

def iif(x0, y0, x1, y1, A):
    return A[x1,y1] - A[x1, y0 - 1] - A[x0 - 1, y1] + A[x0 - 1, y0 - 1]

strawberries_red = cv2.imread('strawberries-binary.pbm', cv2.IMREAD_GRAYSCALE)

#Read the image in and scale it to have “white” values 1 for the strawberry pixels and “black” values 0 for background pixels
binary_strawberries = strawberries_red / 255

#print(iif(0,1, binary_strawberries))
def get_max(A):
    table = summed_area_table(A)
    #result = np.zeroes((A.shape[0] - 100, A.shape[1] - 100))
    max = 0
    max_pos = (0,0)
    for x in range(A.shape[1] - 100):
        for y in range(A.shape[0] - 100):
            area_sum = iif(x, y, x + 100, y + 100, table)
            #result[x,y] = area_sum
            if area_sum > max:
                max = area_sum
                max_pos = (x,y)
    
    return max_pos

max_pos = get_max(strawberries_red)

strawberries_color = cv2.imread('strawberries_color.jpg')
rectangled = cv2.rectangle(strawberries_color, (max_pos[1], max_pos[0]), (max_pos[1]+100, max_pos[0]+100), (0,0,255), 3)

cv2.imwrite('max_mansikkuus.jpg', rectangled)
cv2.waitKey()



#We try to find airport runways in image marion_airport.tiff
#Use OpenCV’s Canny edge detector function to find edges in the image
#edges relating to the runways should be found as much as possible and the others as little as possible
#experiment with the parameter values to get a good result
#include in the report some different outcomes

def add_edges(edges, airport_rgb):
    new = np.copy(edges)
    for x in range(edges.shape[0]):
        for y in range(edges.shape[1]):
            if np.all(edges[x,y] != 0):
                edges[x,y] = [0,255,0]
                new[x,y] = [0, 0, 255]
            else:
                new[x,y] = airport_rgb[x,y]
    return new

airport = cv2.imread('marion_airport.tiff', cv2.IMREAD_GRAYSCALE)
airport = np.uint8(airport)

canny1 = cv2.Canny(airport, 100, 200)
edges1 = cv2.cvtColor(canny1, cv2.COLOR_GRAY2RGB)
canny2 = cv2.Canny(airport, 200, 300)
edges2 = cv2.cvtColor(canny2, cv2.COLOR_GRAY2RGB)
canny3 = cv2.Canny(airport, 400, 500)
edges3 = cv2.cvtColor(canny3, cv2.COLOR_GRAY2RGB)
canny4 = cv2.Canny(airport, 500, 600)
edges4 = cv2.cvtColor(canny4, cv2.COLOR_GRAY2RGB)

airport_rgb = cv2.cvtColor(airport, cv2.COLOR_GRAY2RGB)
canny1out = add_edges(edges1, airport_rgb)
canny2out = add_edges(edges2, airport_rgb)
canny3out = add_edges(edges3, airport_rgb)
canny4out = add_edges(edges4, airport_rgb)

cv2.imwrite('canny1.png', canny1out)
cv2.imwrite('canny2.png', canny2out)
cv2.imwrite('canny3.png', canny3out)
cv2.imwrite('canny4.png', canny4out)


#Use OpenCV’s Hough line transform function to detect the runway pixels
#experiment with the parameter values to get a good result
#include in the report some different outcomes

def houghify(minLineLength, maxLineGap, threshold, canny, A):
    image = np.copy(A)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold, minLineLength, maxLineGap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return image

#minLineLength = 1000
#maxLineGap = 1
#threshold = 40

cv2.imwrite('hough-100-5-50.jpg', houghify(100, 5, 50, canny3, airport))
cv2.imwrite('hough-10-10-10.jpg', houghify(10, 10, 10, canny3, airport))
cv2.imwrite('hough-1000-27-100.jpg', houghify(1000, 27, 100, canny3, airport))
cv2.imwrite('hough-1000-1-100.jpg', houghify(1000, 1, 100, canny3, airport))

#cv2.imwrite('hough.jpg', airport)