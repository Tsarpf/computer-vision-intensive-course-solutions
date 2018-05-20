import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import randint
from scipy import ndimage as nd
from scipy.spatial import distance

print(cv2.__version__)

#We study skeletonization and graph forming using the bear.pbm image
#Load the image in as a grayscale image
#Use OpenCV’s distanceTransform() function on the image (scale the values properly to [0,1]), explain what it does and what are its parameters
#Create the object skeleton with OpenCV’s ximgproc.thinning() function from the original image and try if different values for the thinningType argument would make any difference

bear = cv2.imread('bear.pbm', cv2.IMREAD_GRAYSCALE) / 255

#cv2.imwrite("bear.jpg", bear * 255)
bear_uint8 = np.array(bear * 255, dtype=np.uint8)
#bear= cv2.cvtColor(original_uint8,cv2.COLOR_GRAY2BGR)

#print(bear)
#bear_dist_l2 = cv2.distanceTransform(bear_uint8, cv2.DIST_L2, 3)
bear_dist_l2 = cv2.distanceTransform(bear_uint8, cv2.DIST_C, 3)

max_val = np.amax(bear_dist_l2)
print(max_val)
distanced = (bear_dist_l2 / max_val)

#cv2.imwrite('distanced_bear.jpg', distanced * 255)
#cv2.waitKey()

#cv2.waitKey()

distanced_uint8 = np.array(distanced * 255, dtype=np.uint8)
thinned = cv2.ximgproc.thinning(bear_uint8, thinningType=1)


#kernel =    [[1, 1, 1],
#             [1, -8, 1],
#             [1, 1, 1]]
#cv2.filter2D(thinned, cv2.CV= kernel)
#cv2.imwrite('thinned_bear.jpg', thinned)

#cv2.imshow('derp', thinned)
#cv2.waitKey()


# Implement an algorithm that examines the pixels of the skeleton and creates a graph in which the skeleton’s terminal and branch pixels are nodes and and all other skeleton pixels are replaced with arcs

#def count_adjacent_pixels(pixel):



def mark_terminal(graph, y, x):
    cv2.circle(graph, (x,y), 5, (0, 0, 255))

def mark_branch(graph, y, x):
    cv2.circle(graph, (x,y), 5, (0, 255, 0))

def find_terminals_and_branches(image, graph):
    terminals = []
    branches = []
    arr = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    arr[1:-1,1:-1] = image
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if image[y,x] == 0: continue
            neighbours = 0
            if image[y+1, x+1] != 0: neighbours += 1
            if image[y+1, x] != 0: neighbours += 1
            if image[y+1, x-1] != 0: neighbours += 1
            if image[y, x-1] != 0: neighbours += 1
            if image[y, x+1] != 0: neighbours += 1
            if image[y-1, x+1] != 0: neighbours += 1
            if image[y-1, x] != 0: neighbours += 1
            if image[y-1, x-1] != 0: neighbours += 1
            if neighbours > 2: 
                mark_branch(graph, y, x)
                branches.append((y, x))
            if neighbours == 1: 
                mark_terminal(graph, y , x)
                terminals.append((y, x))
    return terminals, branches

empty_mask = np.array([[1,  1, 1],
              [1,  0, 1],
              [1,  1, 1]])

def mask_dir(dir):
    new = np.copy(empty_mask)
    new[-dir[0], -dir[1]] = 0
    return new

def follow_line(endpoints, mask, cur, points):
    #if cur in endpoints or points[cur[0], cur[1]] == 0:
    #print('cur', cur)
    #print('points', points[cur[0], cur[1]])
    if cur in endpoints:
        return [cur]
    ends = []
    if points[cur[0], cur[1]] == 0:
        return ends 
    points[cur[0], cur[1]] = 0
    for y, row in enumerate(mask):
        for x, cell in enumerate(row):
            if cell == 1:
                ends.extend(follow_line(endpoints, mask_dir((y, x)), (cur[0] + y, cur[1] + x), points))
    return ends


graph = np.zeros((thinned.shape[0], thinned.shape[1], 3), np.uint8)
terminals, branches = find_terminals_and_branches(thinned, graph)

print(terminals[0])
color_thinned = cv2.cvtColor(thinned, cv2.COLOR_GRAY2RGB)
search_img = np.copy(thinned)
nodes = np.concatenate((terminals, branches))
print('nodes', nodes)
for node in nodes:
    results = []
    search_img = np.copy(thinned)
    for y, row in enumerate(empty_mask):
        for x, cell in enumerate(row):
            if cell == 1:
                #print('nodey', node[0] + y)
                #print('nodex', node[1] + x)
                results.extend(follow_line(terminals, mask_dir((y, x)), (node[0] + y, node[1] + x), search_img))
    print('results', results)
    for result in results:
        print('result', terminals[0], result)
        cv2.line(color_thinned, (node[1], node[0]), (result[1], result[0]), (0,0,255), 1)



#cv2.line(color_thinned, (69, 286), (169,186), (0,0,255), 1)
cv2.imshow('derp.jpg', cv2.resize(color_thinned, None, fx=4, fy=4, interpolation=cv2.INTER_AREA))
cv2.waitKey()