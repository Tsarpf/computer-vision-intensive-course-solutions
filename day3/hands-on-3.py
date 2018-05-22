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
bear_uint8 = np.array(bear * 255, dtype=np.uint8)

#cv2.imwrite("bear.jpg", bear * 255)
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


# Implement an algorithm that examines the pixels of the skeleton and creates a graph in which the skeleton’s terminal and branch pixels are nodes and and all other skeleton pixels are replaced with arcs
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
    dir0 = -dir[0] + 1
    dir1 = -dir[1] + 1
    new[dir0, dir1] = 0
    return new

def contains(list, tuple):
    for val in list:
        if val[0] == tuple[0] and val[1] == tuple[1]:
            return True
    return False

def follow_line(endpoints, mask, cur, points):
    ends = []
    if cur[0] >= points.shape[0] or cur[1] >= points.shape[1] or cur[0] < 0 or cur[1] < 0:
        return ends 
    if points[cur[0], cur[1]] == 0:
        return ends 
    if contains(endpoints, cur):
        return [cur]
    points[cur[0], cur[1]] = 0
    for y, row in enumerate(mask):
        for x, cell in enumerate(row):
            if cell == 1:
                idx = x - 1
                idy = y - 1
                ends.extend(follow_line(endpoints, mask_dir((idy, idx)), (cur[0] + idy, cur[1] + idx), points))
    return ends

def graph_skeleton(skeleton):
    #graph = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), np.uint8)
    color_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    testink = np.empty_like(color_skeleton)
    #terminals, branches = find_terminals_and_branches(skeleton, color_skeleton)
    terminals, branches = find_terminals_and_branches(skeleton, testink)

    search_img = np.copy(skeleton)
    if len(terminals) < 1:
        nodes = branches
    elif len(branches) < 1:
        nodes = terminals
    else:
        nodes = np.concatenate((terminals, branches))
    for node in nodes:
        results = []
        #results.extend(follow_line(nodes, empty_mask, node, search_img))
        for y, row in enumerate(empty_mask):
            for x, cell in enumerate(row):
                if cell == 1:
                    idx = x - 1
                    idy = y - 1
                    results.extend(follow_line(nodes, mask_dir((idy, idx)), (node[0] + idy, node[1] + idx), search_img))
        for result in results:
            cv2.line(testink, (node[1], node[0]), (result[1], result[0]), (255,0,0), 1)
    return testink
    #return color_skeleton

#color_skeleton = graph_skeleton(thinned)


#cv2.imshow('derp.jpg', cv2.resize(color_skeleton, None, fx=4, fy=4, interpolation=cv2.INTER_AREA))
#cv2.imwrite('graphed_bear.jpg', cv2.resize(color_skeleton, None, fx=4, fy=4, interpolation=cv2.INTER_AREA))

#
#da_vinci = cv2.imread('da_vinci_vitruve.jpg',0)
#da_vinci_uint8 = np.array(da_vinci * 255, dtype=np.uint8)
#blur = cv2.GaussianBlur(da_vinci_uint8,(5,5),0)
#da_vinci_thinned = cv2.ximgproc.thinning(blur, thinningType=1)
#davinci_graph = graph_skeleton(da_vinci_thinned)
##cv2.imshow('adsfads', davinci_graph)
#cv2.imwrite('vitruve.jpg', davinci_graph)


#sacred = cv2.imread('woman_with_umbrella.png',0)
#sacred_uint8 = np.array(sacred * 255, dtype=np.uint8)
#sacred_blur = cv2.GaussianBlur(sacred_uint8,(5,5),0)
#sacred_thinned = cv2.ximgproc.thinning(sacred_blur, thinningType=1)
#sacred_graph = graph_skeleton(sacred_thinned)

umbrella = cv2.imread('woman_with_umbrella.png',0)
umbrella_uint8 = np.array(umbrella * 255, dtype=np.uint8)
umbrella_blur = cv2.GaussianBlur(umbrella_uint8,(3,3),0)
umbrella_thinned = cv2.ximgproc.thinning(umbrella_blur, thinningType=1)
umbrella_graph = graph_skeleton(umbrella_thinned)

#cv2.imshow('adsfads', sacred_graph)
#cv2.imwrite('umbrella.png', sacred_graph)
cv2.imshow('adsfads', umbrella_graph)
cv2.imwrite('umbrella.png', umbrella_graph)
cv2.waitKey()