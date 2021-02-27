
import cv2
import numpy as np
import os



img_p1 = "../data/manual/campus/campus1.jpg"
img_p2 = "../data/manual/campus/campus2.jpg"
img1 = cv2.imread(img_p1)
img2 = cv2.imread(img_p2)
# cv2.imshow("image1",img1)
# cv2.imshow("image2",img2)
# cv2.waitKey(0)
img1_points = np.float32([[126,313],[89,64],[176,47],[122,49],[30,111]])
img2_points = np.float32([[392,318],[347,71],[432,47],[379,53],[290,121]])
matrix,_ = cv2.findHomography(img1_points, img2_points)
offset = np.array([[ 1 , 0 , 50],
[ 0 , 1 , 50],
[ 0 , 0 ,    1    ]])
matrix = offset@matrix
dst = cv2.warpPerspective(img1,matrix,(960, 800))
# cv2.imshow("fnal",dst)
# cv2.waitKey(0)
dst[50:img2.shape[0]+50, 50:img2.shape[1]+50] = img2
resized = cv2.resize(dst, (480,400) , interpolation = cv2.INTER_AREA)
cv2.imshow("fnal",resized)
cv2.waitKey(0)
