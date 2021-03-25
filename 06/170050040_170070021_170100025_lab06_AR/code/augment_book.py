#!/usr/bin/env python


import argparse
import cv2
import sys, os
import numpy as np
import math 

parser = argparse.ArgumentParser()
parser.add_argument('-w', default=1.5)
parser.add_argument('-l', default=5)
parser.add_argument('-b', default=3)
parser.add_argument('-x', default=60)
parser.add_argument('-y', default=40)
parser.add_argument('-theta', default=45)

args = parser.parse_args()

w = float(args.w)
l = float(args.l)
b = float(args.b)
x = int(args.x)
y = int(args.y)
theta = int(args.theta)
theta_rad = (theta*math.pi)/180
def draw_book_on_img(image, projects):
    projects = np.array(projects,np.int32)
    projects = projects.reshape(8,1,2)
    image = cv2.drawContours(image, [projects[:4]], -1, (0, 0, 255), 5)
    print(projects[1])
    for i in range(4):
        image = cv2.line(image, tuple(projects[i][0]), tuple(projects[i+4][0]), (0, 0, 255), 5)
    image = cv2.drawContours(image, [projects[4:]], -1, (0, 0, 255), 5)
 
    return image

def get_data():
  directory = os.path.join(sys.path[0], "../data/augment_book/")
  images = [f for f in os.listdir(directory) if 
          os.path.isfile(os.path.join(directory,f))]
  images.sort()
  print(images)
  NUMBER_OF_WALL_IMAGES = 6
  front_path = os.path.join(directory, images[NUMBER_OF_WALL_IMAGES])
  front = cv2.imread(front_path)
  side_path = os.path.join(directory, images[NUMBER_OF_WALL_IMAGES+2])
  side = cv2.imread(side_path)
  wall_path = os.path.join(directory, images[0])
  wall = cv2.imread(wall_path)
  print(front_path,"gggggg")
  # cv2.imshow("front", front)
  # cv2.imshow("side", side)
  # cv2.waitKey(0)
  return images
img1_points = []
img2_points = []
img3_points = []
img4_points = []
img5_points = []
img6_points = []

imm = get_data()
img = []
img.append(cv2.imread("../data/augment_book/"+imm[0]))
img.append(cv2.imread("../data/augment_book/"+imm[1]))
img.append(cv2.imread("../data/augment_book/"+imm[2]))
img.append(cv2.imread("../data/augment_book/"+imm[3]))
img.append(cv2.imread("../data/augment_book/"+imm[4]))
img.append(cv2.imread("../data/augment_book/"+imm[5]))

res = img[0].shape[0]/600

def coordinates_click(event, x, y, flags, params): 
  
    if event == cv2.EVENT_LBUTTONDOWN and params=="image1": 
        img1_points.append([x,y])
    elif event == cv2.EVENT_LBUTTONDOWN and params=="image2":
        img2_points.append([x,y]) 
    elif event == cv2.EVENT_LBUTTONDOWN and params=="image3": 
        img3_points.append([x,y])
    elif event == cv2.EVENT_LBUTTONDOWN and params=="image4":
        img4_points.append([x,y]) 
    elif event == cv2.EVENT_LBUTTONDOWN and params=="image5": 
        img5_points.append([x,y])
    elif event == cv2.EVENT_LBUTTONDOWN and params=="image6":
        img6_points.append([x,y])                    


# cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
# cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
# cv2.namedWindow('image3',cv2.WINDOW_NORMAL)
# cv2.namedWindow('image4',cv2.WINDOW_NORMAL)
# cv2.namedWindow('image5',cv2.WINDOW_NORMAL)
# cv2.namedWindow('image6',cv2.WINDOW_NORMAL)

# cv2.resizeWindow('image1', (min(int(img[0].shape[1]/res),600),600))
# cv2.resizeWindow('image2', (min(int(img1.shape[1]/res),600),600))
# cv2.resizeWindow('image3', (min(int(img1.shape[1]/res),600),600))
# cv2.resizeWindow('image4', (min(int(img1.shape[1]/res),600),600))
# cv2.resizeWindow('image5', (min(int(img1.shape[1]/res),600),600))
# cv2.resizeWindow('image6', (min(int(img1.shape[1]/res),600),600))

cv2.imshow('image1', img[0])
# cv2.imshow('image2', img2)
# cv2.imshow('image3', img3)
# cv2.imshow('image4', img4)
# cv2.imshow('image5', img5)
# cv2.imshow('image6', img6)

# cv2.setMouseCallback('image1', coordinates_click, "image1") 
# cv2.setMouseCallback('image2', coordinates_click, "image2") 
# cv2.setMouseCallback('image3', coordinates_click, "image3") 
# cv2.setMouseCallback('image4', coordinates_click, "image4") 
# cv2.setMouseCallback('image5', coordinates_click, "image5") 
# cv2.setMouseCallback('image6', coordinates_click, "image6") 

cv2.waitKey(0)

pts = np.array([[25, 70], [25, 160], 
                [110, 200], [200, 160], 
                [200, 70], [110, 20]],
               np.int32)
  
# pts = pts.reshape((-1, 1, 2))
# print([pts])
# img[1] = np.array(img[1],dtype = np.float32)
# imgmm = cv2.polylines(img[1], [pts], 
#                       True, (0,254,255), 10)


print(img1_points,img2_points,img3_points,img4_points,img5_points,img6_points)
img_points = np.array([[[692, 2988], [960, 2955], [1185, 2921], [651, 2696], [901, 2679], [1135, 2654]],
 [[358, 2629], [576, 2671], [776, 2713], [434, 2429], [642, 2479], [851, 2512]],
  [[567, 3038], [834, 3021], [1076, 3005], [559, 2779], [809, 2771], [1051, 2754]],
   [[818, 3055], [1043, 2980], [1243, 2913], [734, 2829], [960, 2771], [1160, 2704]],
    [[417, 2579], [626, 2671], [818, 2746], [525, 2404], [709, 2496], [901, 2579]],
     [[567, 2829], [759, 2821], [934, 2804], [567, 2637], [751, 2629], [926, 2612]]],dtype = np.float32)
print(img_points.shape)   
book_points_old = np.float32([[0, 0, 0], [0,b,0],[l,b,0],[l,0,0],[0,0,w],[0,b,w],[l,b,w],[l,0,w]])
real_world_points = np.array([[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]]],dtype = np.float32)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_world_points, img_points, img[0].shape[1::-1], None, None)

print(mtx)
mtx = np.array(mtx)


rt_matrix = [[math.cos(theta_rad),-math.sin(theta_rad),x],
            [math.sin(theta_rad),math.cos(theta_rad),y],
            [0,0,1]]
book_temp = book_points_old.copy()
book_temp[:,2] = 1   
book_temp[:,0] = book_temp[:,0] - x
book_temp[:,1] = book_temp[:,1] - y

print(book_temp)
# print(book_points) 
        
new_book = rt_matrix@np.transpose(book_temp)
book_points = book_points_old.copy()
book_points[:,:2] = np.transpose(new_book)[:,:2]
print(book_points)
print(new_book)
# new_book[]
# project 3D points to image plane
for p in range(len(img_points)):
    ret,rvecs, tvecs = cv2.solvePnP(real_world_points[p], img_points[p], mtx, None)
    projects, jac = cv2.projectPoints(book_points_old, rvecs, tvecs, mtx, None)
    print(projects,"ggggggg") 
    im = draw_book_on_img(img[p], projects)
    cv2.namedWindow('gg',cv2.WINDOW_NORMAL)

    cv2.resizeWindow('gg', (min(int(img[p].shape[1]/res),600),600))
    cv2.imshow("gg",im)
    cv2.waitKey(0)

  