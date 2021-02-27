
import cv2
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join
images = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
images.sort()
img_p1 = sys.argv[1] + "/" + images[0]
img_p2 = sys.argv[1] + "/" + images[1]
img1 = cv2.imread(img_p1)
img2 = cv2.imread(img_p2)
res = img1.shape[0]/480

resized1 = cv2.resize(img1, (int(img1.shape[1]/res),480) , interpolation = cv2.INTER_AREA)
resized2 = cv2.resize(img2, (int(img2.shape[1]/res),480) , interpolation = cv2.INTER_AREA)


#campus
# img2_points = np.float32([[126,313],[89,64],[176,47],[122,49],[30,111]])
# img1_points = np.float32([[392,318],[347,71],[432,47],[379,53],[290,121]])
#society
img1_points = res*np.float32([[140,103],[140,390],[192,130],[160,116],[158,153],[216,135]])
img2_points = res*np.float32([[39,86],[35,378],[91,119],[60,102],[57,140],[112,126]])
#yard
# img2_points = res*np.float32([[338,266],[366,143],[131,297],[286,137],[193,237]])
# img1_points = res*np.float32([[549,275],[573,144],[342,297],[487,143],[397,243]])
#gate
# img1_points = res*np.float32([[161,46],[136,61],[132,384],[131,145],[183,254],[217,258],[203,63],[213,59],[122,56],[116,173],[212,249]])
# img2_points = res*np.float32([[45,64],[27,177],[35,404],[22,160],[81,267],[113,268],[87,83],[97,80],[7,70],[7,190],[107,261]])
#House
# img1_points = res*np.float32([[188,216],[215,337],[157,254],[197,230],[205,215]])
# img2_points = res*np.float32([[50,211],[75,327],[20,250],[60,223],[67,210]])


matrix,_ = cv2.findHomography(img2_points, img1_points)
offset = np.array([[ 1 , 0 , 0],
[ 0 , 1 , int(img1.shape[0]/2)],
[ 0 , 0 ,    1    ]])
matrix = offset@matrix
dst = cv2.warpPerspective(img2,matrix,(img1.shape[1]+img2.shape[1], int(img1.shape[0]*2)))
res_final = dst.shape[0]/400
dst2 = dst.copy()

dst2[int(img1.shape[0]/2):img1.shape[0]+int(img1.shape[0]/2), :img1.shape[1]] = img1
resized = cv2.resize(dst2, (int(dst.shape[1]/res_final), 400) , interpolation = cv2.INTER_AREA)
cv2.imshow("Final",resized)
cv2.waitKey(0)
