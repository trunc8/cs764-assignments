import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "../data/calib_images/"
img1_points = []
img2_points = []

    
# def coordinates_click(event, x, y, flags, params): 
  
#     if event == cv2.EVENT_LBUTTONDOWN and params=="img1": 
#         img1_points.append([x,y])
#     elif event == cv2.EVENT_LBUTTONDOWN and params=="img2":
#         img2_points.append([x,y]) 

img1 = cv2.imread(path+"1.jpg")
img2 = cv2.imread(path+"2.jpg")
res = img1.shape[0]/600

# cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
# cv2.namedWindow('img2',cv2.WINDOW_NORMAL)

# cv2.resizeWindow('img1', (min(int(img1.shape[1]/res),600),600))
# cv2.resizeWindow('img2', (min(int(img2.shape[1]/res),600),600))

# cv2.imshow("img1",img1)
# cv2.imshow("img2",img2)

# cv2.setMouseCallback('img1', coordinates_click, "img1") 
# cv2.setMouseCallback('img2', coordinates_click, "img2")

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img1_points,img2_points)
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img1_points = np.asarray(img1_points,dtype = np.float32)
objpoints = np.array([[[3,0,0],[6,0,0],[9,0,0],[3,3,0],[6,3,0],[9,3,0]],[[3,0,0],[6,0,0],[9,0,0],[3,3,0],[6,3,0],[9,3,0]]],dtype = np.float32)
# print(type(objpoints))

# img_points = img1_points.append(img2_points)
objpoints = np.array([[[3,0,0],[6,0,0],[9,0,0],[3,3,0],[6,3,0],[9,3,0]],[[3,0,0],[6,0,0],[9,0,0],[3,3,0],[6,3,0],[9,3,0]]],dtype = np.float32)
img_points = np.array([[[960, 2955], [1185, 2921], [1435, 2880], [909, 2679], [1135, 2654], [1377, 2629]],[[576, 2671], [776, 2704], [1018, 2754], [642, 2479], [851, 2512], [1068, 2562]]],dtype = np.float32)

print(img_points)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, img_points, img2.shape[1::-1], None, None)
print(mtx)