import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", action="store", dest="filepath")
args = parser.parse_args()


# Function to rearrange points in the order we need
# to be consistent with order in objpoints array
def rearrange(l):
    l1 = []
    l2 = []
    l3 = []
    maxx = 0
    minx = 10000
    for i in range(6):
        if (maxx<l[i][0]):
            maxx = l[i][0]
        elif (minx > l[i][0]):
            minx = l[i][0]
        else: 
            pass
    for i in range(6):
        if (abs(l[i][0]-maxx)<5):
            l3.append(l[i])
        elif (abs(minx - l[i][0])<5):
            l1.append(l[i])
        else:
            l2.append(l[i])
    if (l1[0][1] > l1[1][1]):
        l[5] = l1[0]
        l[2] = l1[1]
    else:
        l[5] = l1[1]
        l[2] = l1[0]
    if (l2[0][1] > l2[1][1]):
        l[4] = l2[0]
        l[1] = l2[1]
    else:
        l[4] = l2[1]
        l[1] = l2[0]
    if (l3[0][1] > l3[1][1]):
        l[3] = l3[0]
        l[0] = l3[1]
    else:
        l[3] = l3[1]
        l[0] = l3[0]
    return l


f = open(args.filepath,'r')
pts = f.readlines()
for i in range(len(pts)):
    pts[i] = pts[i][:-1]
    pts[i] = pts[i].split(' ')
    pts[i][0] = float(pts[i][0])
    pts[i][1] = float(pts[i][1])

pts1 = pts[:6]
pts2 = pts[6:12]
pm1 = rearrange(pts1)
pm2 = rearrange(pts2)

print("Rearranged points for image 1 are")
print(pm1)
print("Rearranged points for image 2 are")
print(pm2)

objpoints = np.array([[[0, 0, 0], [3, 0, 0], [6, 0, 0], [0, 3, 0], [3, 3, 0], [6, 3, 0]],
                      [[0, 0, 0], [3, 0, 0], [6, 0, 0], [0, 3, 0], [3, 3, 0], [6, 3, 0]]], dtype = np.float32)

# Image points in sequence corresponding to above
imgpoints = np.array([pm1, pm2], dtype=np.float32)

print (objpoints.shape)
print (imgpoints.shape)
exit()
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints,
                        (720, 1280), None, None, flags=
                         (cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2
                          + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6))

print("Learned/Fitted Intrinsic matrix is")
print(mtx)
print("Rotation vectors are")
print(rvecs)
print("Translation vectors are")
print(tvecs)

projected0, _ = cv2.projectPoints(objpoints[0], rvecs[0], tvecs[0], mtx, dist)
projected1, _ = cv2.projectPoints(objpoints[1], rvecs[1], tvecs[1], mtx, dist)

# Reproject points
projected0 = projected0.reshape(1, 6, 2)
projected1 = projected1.reshape(1, 6, 2)
error0 = np.sqrt(np.sum(np.square(imgpoints[0]-projected0)))/12
error1 = np.sqrt(np.sum(np.square(imgpoints[1]-projected1)))/12

print("Average projection error {0:.2f}".format(error0+error1))
