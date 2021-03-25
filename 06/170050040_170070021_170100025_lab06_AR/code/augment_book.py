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
parser.add_argument('-x', default=1000)
parser.add_argument('-y', default=1000)
parser.add_argument('-theta', default=1000)

args = parser.parse_args()

w = float(args.w)
l = float(args.l)
b = float(args.b)
x = int(args.x)
y = int(args.y)
theta = int(args.theta)
theta_rad = (theta*math.pi)/180

def draw_book_on_img(image, projects):
    centre_old = (projects[0] +projects[2])/2
    
    new_x = (x*image.shape[1])/100
    new_y = (y*image.shape[0])/100
    distx = new_x - centre_old[0][0]
    disty = new_y - centre_old[0][1]
  
    new_projects = projects.copy()
    new_projects[:,:,0] = projects[:,:,0] + distx * (x!=1000)
    new_projects[:,:,1] = projects[:,:,1] + disty * (y!=1000)
   
    
    new_projects = np.array(new_projects,np.int32)
    new_projects = new_projects.reshape(8,1,2)
    image = cv2.drawContours(image, [new_projects[:4]], -1, (0, 0, 255), 5)
    for i in range(4):
        image = cv2.line(image, tuple(new_projects[i][0]), tuple(new_projects[i+4][0]), (0, 0, 255), 5)
    image = cv2.drawContours(image, [new_projects[4:]], -1, (0, 0, 255), 5)
 
    return image

def get_data():
  directory = os.path.join(sys.path[0], "../data/augment_book/")
  images = [f for f in os.listdir(directory) if 
          os.path.isfile(os.path.join(directory,f))]
  images.sort()
  NUMBER_OF_WALL_IMAGES = 6
  images = []
  for i in range(NUMBER_OF_WALL_IMAGES):
    wall_path = os.path.join(directory, images[0])
    wall = cv2.imread(wall_path)
    images.append(wall)
  front_path = os.path.join(directory, images[NUMBER_OF_WALL_IMAGES])
  front = cv2.imread(front_path)
  images.append(front)  
  side_path = os.path.join(directory, images[NUMBER_OF_WALL_IMAGES+2])
  side = cv2.imread(side_path)
  images.append(side)  
  return images

imm = get_data()
res = imm[0].shape[0]/600

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

img_points = np.array([[[692, 2988], [960, 2955], [1185, 2921], [651, 2696], [901, 2679], [1135, 2654]],
 [[734, 3005], [934, 2946], [1118, 2921], [701, 2838], [909, 2796], [1101, 2754]],
  [[567, 3038], [834, 3021], [1076, 3005], [559, 2779], [809, 2771], [1051, 2754]],
   [[818, 3055], [1043, 2980], [1243, 2913], [734, 2829], [960, 2771], [1160, 2704]],
    [[417, 2579], [626, 2671], [818, 2746], [525, 2404], [709, 2496], [901, 2579]],
     [[567, 2829], [759, 2821], [934, 2804], [567, 2637], [751, 2629], [926, 2612]]],dtype = np.float32)

book_points_old = np.float32([[0, 0, 0], [0,b,0],[l,b,0],[l,0,0],[0,0,w],[0,b,w],[l,b,w],[l,0,w]])

real_world_points = np.array([[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]]],dtype = np.float32)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_world_points, img_points, img[0].shape[1::-1], None, None)
mtx = np.array(mtx)

rt_matrix = [[math.cos(theta_rad),-math.sin(theta_rad),l/2],
            [math.sin(theta_rad),math.cos(theta_rad),b/2],
            [0,0,1]]

book_temp = book_points_old.copy()
book_temp[:,2] = 1   
book_temp[:,0] = book_temp[:,0] - l/2
book_temp[:,1] = book_temp[:,1] - b/2

        
new_book = rt_matrix@np.transpose(book_temp)
book_points = book_points_old.copy()
book_points[:,:2] = np.transpose(new_book)[:,:2]


book_points_old =  book_points*(theta!=1000) + book_points_old*(theta==1000)
for p in range(len(img_points)):
    ret,rvecs, tvecs = cv2.solvePnP(real_world_points[p], img_points[p], mtx, None)
    projects, jac = cv2.projectPoints(book_points_old, rvecs, tvecs, mtx, None)
    im = draw_book_on_img(img[p], projects)
    cv2.namedWindow('gg',cv2.WINDOW_NORMAL)

    cv2.resizeWindow('gg', (min(int(img[p].shape[1]/res),600),600))
    cv2.imshow("gg",im)
    cv2.waitKey(0)

  
