import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage as nd
from scipy.spatial import distance

print(cv2.__version__)

#We will study the use of OpenCV’s versatile and powerful CascadeClassifier with people.jpeg image and other examples
#Start with the sample code from Face Detection using Haar Cascades tutorial
#You will need to link the model files from /usr/share/opencv/haarcascades/ or use them directly from there
#Experiment face and eye detection with the default code and people.jpeg image trying to make the eye detection work better by modifying the parameters of the method
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
img = cv2.imread('people.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.01, 250)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.001, 5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imwrite('eyeglasses_detect.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Find the actual C/C++ source code of the CascadeClassifier class to see how OpenCV has been implemented and documented! Report the file name of the source code….

#There are various other cascade classifier models in the same directory, experiment with a couple of them by using images suitable for each purpose

#Remember that failure cases are more interesting than successful cases!
#
#Report also, how long it took for you to complete the home work
