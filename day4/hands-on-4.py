import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage as nd
from scipy.spatial import distance

print(cv2.__version__)

#bear = cv2.imread('bear.pbm', cv2.IMREAD_GRAYSCALE) / 255
#cv2.imwrite("bear.jpg", bear * 255)
#bear_uint8 = np.array(bear * 255, dtype=np.uint8)
#bear= cv2.cvtColor(original_uint8,cv2.COLOR_GRAY2BGR)



#We study keypoint detection and description with images poster.jpeg and frame.jpeg

#The detector--descriptor framework together with nearest neighbor search allows matching corresponding points in two images

#ORB detector and descriptor is available in OpenCV whereas SIFT and SURF might be better but they are only available if opencv_contrib is included (not in Ubuntu)

#Load the images in
poster = cv2.imread('poster.jpeg', cv2.IMREAD_GRAYSCALE)
#poster = cv2.imread('poster.jpeg', cv2.IMREAD_GRAYSCALE) / 255
frame = cv2.imread('frame.jpeg', cv2.IMREAD_GRAYSCALE)

#Use OpenCV’s cv2.ORB_create() to create the detector and descriptor object orb
#Apply orb.detectAndCompute() to get the keypoints and their corresponding descriptors in both images
orb = cv2.ORB_create()
poster_kp, poster_des = orb.detectAndCompute(poster, None)
frame_kp, frame_des = orb.detectAndCompute(frame, None)

#plt.imshow(poster2), plt.show()
#plt.imshow(frame2), plt.show()

#ORB returns by default 500 keypoints, which might be a good number, could be changed

#Visualize the detections using cv2.drawKeypoints()
poster2 = cv2.drawKeypoints(poster, poster_kp, None, color=(0, 255, 0), flags=0)
frame2 = cv2.drawKeypoints(frame, frame_kp, None, color=(0, 255, 0), flags=0)

poster2_resize = cv2.copyMakeBorder(poster2, 0, frame2.shape[0] - poster2.shape[0], 0, 0, cv2.BORDER_CONSTANT)
#Combine the two images side by side into one large one
combod = np.concatenate((frame2, poster2_resize), axis=1)

#Create a nearest neighbor search model with cv2.ml.KNearest_create() etc., by using descriptors from the poster image, use keypoint indices as the feature vector labels

knn = cv2.ml.KNearest_create()
poster_des_f32 = np.asarray(poster_des, dtype=np.float32)
frame_des_f32 = np.asarray(frame_des, dtype=np.float32)
poster_kp_vec = cv2.KeyPoint_convert(poster_kp)
frame_kp_vec = cv2.KeyPoint_convert(frame_kp)

labels = np.asarray(np.arange(0, len(poster_kp)), dtype=np.float32).reshape(500, 1)
knn.train(poster_des_f32, cv2.ml.ROW_SAMPLE, labels)
ret, results, neighbours, dist = knn.findNearest(frame_des_f32, k=1)

### Connect the matching keypoint pairs across the images
#combod = np.concatenate((frame2, poster2_resize), axis=1)
#poster_resize = cv2.copyMakeBorder(poster2, 0, frame2.shape[0] - poster2.shape[0], 0, 0, cv2.BORDER_CONSTANT)
poster_offset = (frame2.shape[1], frame2.shape[0])

# results is a mapping between the original group and the end group
#print(results)

red = (0, 0, 255)
# poster_idx = 0
# for point in results:
#     frame_idx = int(point[0])
#     poster_kp_pos = poster_kp_vec[poster_idx]
#     poster_img_pos = (int(poster_kp_pos[0] + poster_offset[0]), int(poster_kp_pos[1]))
#     frame_img_pos = (int(frame_kp_vec[frame_idx][0]), int(frame_kp_vec[frame_idx][1]))
#     print(poster_img_pos)
#     print(frame_img_pos)
#     cv2.line(combod, poster_img_pos, frame_img_pos, red, 1)
#     poster_idx += 1

print('ses')
#cv2.imwrite('one_way_match_knn.png', combod)

#Show the result
#plt.imshow(combod), plt.show()

#Change to use a 1-NN matching also in the opposite direction by creating a k-NN model from the frame image’s descriptors
knn_ftop = cv2.ml.KNearest_create()
knn_ftop.train(frame_des_f32, cv2.ml.ROW_SAMPLE, labels)
ret, results_ftop, neighbours, dist = knn_ftop.findNearest(poster_des_f32, k=1)

keijo = []
for frame_idx in range(len(results_ftop)):
    poster_idx = int(results_ftop[frame_idx][0])
    back_ref = int(results[poster_idx][0])
    print(frame_idx, poster_idx, back_ref)
    if frame_idx == back_ref:
        #poster_kp_pos = poster_kp_vec[poster_idx]
        #frame_kp_pos = frame_kp_vec[frame_idx]
        #poster_kp_pos = poster_kp_vec[frame_idx]
        #frame_kp_pos = frame_kp_vec[poster_idx]
        #poster_kp_pos = poster_kp_vec[poster_idx]
        #frame_kp_pos = frame_kp_vec[back_ref]
        poster_kp_pos = poster_kp_vec[back_ref]
        frame_kp_pos = frame_kp_vec[poster_idx]
        #poster_kp_pos = poster_kp_vec[frame_idx]
        #frame_kp_pos = frame_kp_vec[back_ref]
        #poster_kp_pos = poster_kp_vec[back_ref]
        #frame_kp_pos = frame_kp_vec[frame_idx]
        poster_img_pos = (int(poster_kp_pos[0] + poster_offset[0]), int(poster_kp_pos[1]))
        frame_img_pos = (int(frame_kp_pos[0]), int(frame_kp_pos[1]))
        cv2.line(combod, poster_img_pos, frame_img_pos, red, 1)
        #keijo.append((poster_idx, frame_idx))

    # loop idx in loop tells start set index, value at that index tells end set index
    # we should compare whether end set index corresponds to start set index
    # aka for first results compare if the other backwards is the same
    # so we should check for each value at an index, whether that index in the other set contains this index.

# [5] --> toisen idx=5 osoittaman pitäisi olla ekan idx
# [17]
# [6]

plt.imshow(combod), plt.show()

#Find the nearest descriptors in the first image for those in the second image, and vice versa

#If a pair of descriptors, one in the first image ant the other in the second image, are reciprocally nearest ones to each other, consider them as a matching pair

#Again connect the matching pairs and show results

#Change to take the frame images from file video.avi (found insize video.zip in Drive)  and record the processing time per frame

#Try if you can change to use FLANN as the one-directional 1-NN matcher and record the speed again

#You can also experiment with SIFT and SURF if you have them: do the results look better and were they slower than ORB?

#Report again how long it took to complete the assignment
