### Task 1
# We will use image strawberries.tiff to study color-based segmentation and morphological closing
### Task 2?
# Select representative BGR sample points corresponding to the red and green colors in the berries
# wat?

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage as nd
from scipy.spatial import distance

strawberries = cv2.imread('strawberries.tiff')
cv2.imwrite('strawberries.jpg', strawberries)

### Task 3
# Define spheres around the reference points and count how many pixels fall in them when increasing the radius from zero until all pixels are inside the spheres
def count_in_sphere(radius, distances):
    count = 0
    for column in distances:
        for distance in column:
            if(distance < radius):
                count += 1
    return count

def calc_distances(center, A):
    compare_color = A[center[0]][center[1]]
    distances = np.full((strawberries.shape[0], strawberries.shape[1]), 0)
    x, y = 0,0
    for column in A:
        for pixel in column:
            distances[y][x] = distance.euclidean(compare_color, pixel)
            x += 1
        x = 0
        y += 1
    return distances


#print('derp ' + str(count_in_sphere((65, 200), 10, strawberries)))
# strawberry red area
#cv2.imshow("red", cv2.resize(strawberries[30:100, 150:250], None, fx=2, fy=2, interpolation=cv2.INTER_AREA))


red_distances = calc_distances((65, 200), strawberries)

red_enabled = False
if red_enabled:
    pixel_count = strawberries.shape[0] * strawberries.shape[1]
    count = 0
    radius = 0
    count_per_radius_red = []
    while count < pixel_count and red_enabled:
        print('ratius ' + str(radius))
        radius += 1
        count = count_in_sphere(radius, red_distances)
        count_per_radius_red.append(count)

    print('doned')
    plt.plot(count_per_radius_red)
    plt.show()
    plt.savefig('pixels_per_radius_red.png')

# strawberry green area 
#cv2.imshow("green", cv2.resize(strawberries[100:180, 280:370], None, fx=2, fy=2, interpolation=cv2.INTER_AREA))
#cv2.waitKey()

green_distances = calc_distances((120, 300), strawberries)

green_enabled = False
if green_enabled:
    pixel_count = strawberries.shape[0] * strawberries.shape[1]
    count = 0
    radius = 0
    count_per_radius_green = []
    while count < pixel_count and green_enabled:
        print('ratius ' + str(radius))
        radius += 1
        count = count_in_sphere(radius, green_distances)
        count_per_radius_green.append(count)

    print('doned')
    plt.plot(count_per_radius_green)
    plt.show()
    plt.savefig('pixels_per_radius_green.png')


### Task 5
#Does it seem possible to select a good threshold value based on those curves?
#yes
### Task 6
# With four different threshold radii, show the red and green binary masks that contain the segmented pixels

def mark_pikkels(radius, distances):
    mask = np.full((distances.shape[0], distances.shape[1]), 0)
    for y in range(len(distances) - 1):
        for x in range(len(distances[y]) - 1):
            if(distances[y][x] < radius):
                mask[y][x] = 1
    return mask

green_mask_45 = mark_pikkels(45, green_distances) * 255
green_mask_70 = mark_pikkels(70, green_distances) * 255
green_mask_90 = mark_pikkels(90, green_distances) * 255
green_mask_110 = mark_pikkels(110, green_distances) * 255

plt.imshow(green_mask_45)
plt.savefig('green_mask_45.png')
plt.imshow(green_mask_70)
plt.savefig('green_mask_70.png')
plt.imshow(green_mask_90)
plt.savefig('green_mask_90.png')
plt.imshow(green_mask_110)
plt.savefig('green_mask_110.png')

red_mask_45 = mark_pikkels(45, red_distances) * 255
red_mask_70 = mark_pikkels(70, red_distances) * 255
red_mask_90 = mark_pikkels(90, red_distances) * 255
red_mask_110 = mark_pikkels(110, red_distances) * 255
plt.imshow(red_mask_45)
plt.savefig('red_mask_45.png')
plt.imshow(red_mask_70)
plt.savefig('red_mask_70.png')
plt.imshow(red_mask_90)
plt.savefig('red_mask_90.png')
plt.imshow(red_mask_110)
plt.savefig('red_mask_110.png')

### Task 7
# Select  a  good  threshold  radius  that  works  for  bothread and green90
# 90 seems to work alright for both

### Task 8
# Create a binary mask image of the segmented red and green pixels
def combine(red, green):
    combined = np.full((red.shape[0], red.shape[1]), 0)
    for y in range(len(red) - 1):
        for x in range(len(red[y]) - 1):
            if red[y][x] == 1 or green[y][x] == 1:
                combined[y][x] = 1
    return combined

combined = combine(red_mask_90/255, green_mask_90/255)

plt.imshow(combined * 255)
plt.savefig('combined_red_green_90.png')