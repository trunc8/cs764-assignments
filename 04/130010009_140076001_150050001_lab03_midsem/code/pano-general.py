import numpy as np
import cv2
import matplotlib.pyplot as plt

import os, sys

# directory = path-to-directory-containing-2-images

if len(sys.argv) < 2:
  # default, if no arguments are passed
  directory = os.path.join(sys.path[0], "../data/general/mountain")
  referenceImage = 3
else:
  directory = sys.argv[1]
  referenceImage = int(sys.argv[2])
  # print(directory)
  # print(referenceImage)

images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
images.sort()

# Query = Reference image
# Train = Transform image

def stitchPairOfImages(query_img, train_img):
  ## Display query and train images (for debugging)
  # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
  # ax1.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
  # ax1.set_title("Reference image", fontsize=14)
  # ax1.axis('off')

  # ax2.imshow(cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB))
  # ax2.set_title("Transform image", fontsize=14)
  # ax2.axis('off')

  # plt.suptitle("Input images", fontsize=18)
  # plt.show()

  ## ORB = Oriented FAST and Rotated BRIEF
  # Initiate ORB detector
  orb = cv2.ORB_create()

  # find the keypoints and descriptors with ORB
  kp1, des1 = orb.detectAndCompute(train_img,None)
  kp2, des2 = orb.detectAndCompute(query_img,None)

  ### REFLECTIONESSAY: Why this order.


  ## BFMatcher = Brute Force Matcher
  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)
  top_matches = matches

  query_pts = np.float32([ kp1[m.queryIdx].pt for m in top_matches])
  train_pts = np.float32([ kp2[m.trainIdx].pt for m in top_matches])

  ## RANSAC = Random Sample Consensus
  H, _ = cv2.findHomography(query_pts, train_pts,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=4)
  # print(H)

  '''
  The result image size needs to be large enough to accomodate the warped
  train_img. So we take the dimensions of the query image
  and call that a block. Imagine the result image to be a 5x5 grid of
  individual blocks. The center block would be the offset position for the
  reference(query) image and also pre-multiplied to H before warping the
  train image.
  '''

  query_height = query_img.shape[0]
  query_width = query_img.shape[1]

  scale = 5
  result_height = scale*query_height
  result_width = scale*query_width

  offset = np.array([[ 1 , 0 , (scale//2)*query_width],
                     [ 0 , 1 , (scale//2)*query_height],
                     [ 0 , 0 ,    1    ]])
  offset_H = offset@H

  mask = np.ones(train_img.shape)
  mask = cv2.warpPerspective(mask, offset_H, (result_width, result_height))
  mask = mask==1

  warped_train_img = cv2.warpPerspective(train_img,
                                         offset_H,
                                         (result_width, result_height)
                                        )
  # Blank base image
  result = np.zeros(warped_train_img.shape, dtype=np.uint8)
  
  ## ALTERNATE 1
  # # Transformed image is inserted using a mask
  # result[mask] = warped_train_img[mask]
  # # Reference image is overlayed unchanged into the center grid
  # result[(scale//2)*query_height+1 : (1+scale//2)*query_height+1,
  #        (scale//2)*query_width+1  : (1+scale//2)*query_width+1] = query_img

  ## ALTERNATE 2
  # Reference image is inserted unchanged into the center grid
  result[(scale//2)*query_height+1 : (1+scale//2)*query_height+1,
         (scale//2)*query_width+1  : (1+scale//2)*query_width+1] = query_img
  # Transformed image is overlayed using a mask
  result[mask] = warped_train_img[mask]


  # In order to crop black borders-
  result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

  contours,hierarchy = cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnt = contours[0]
  x,y,w,h = cv2.boundingRect(cnt)
  cropped_result = result[y:y+h,x:x+w]

  ## Display stitched image (for debugging)
  # plt.figure(figsize=(20,10))
  # plt.title("Panaromic image", fontsize=16)
  # plt.imshow(cv2.cvtColor(cropped_result, cv2.COLOR_BGR2RGB))
  # plt.axis('off')
  # plt.show()

  return cropped_result


def generalParanoma(images, referenceImage):
  num_images = len(images)
  index = referenceImage-1
  if (index >= num_images):
    print("Invalid reference image entered. Defaulting to first image.")
    index = 0
  
  query_img_path = os.path.join(directory, images[index])
  query_img = cv2.imread(query_img_path)
  
  for i in range(1,num_images+1):
    train_img_path = os.path.join(directory, images[(index+i)%num_images])
    train_img = cv2.imread(train_img_path)

    query_img = stitchPairOfImages(query_img, train_img)

  result = query_img
  return result

result = generalParanoma(images, referenceImage)
plt.figure(figsize=(20,10))
plt.title("Panaromic image", fontsize=16)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# query_img_path = os.path.join(directory, images[0])
# train_img_path = os.path.join(directory, images[1])

# query_img = cv2.imread(query_img_path)
# train_img = cv2.imread(train_img_path)

# query_img = stitchPairOfImages(query_img, train_img)

# train_img_path = os.path.join(directory, images[2])
# train_img = cv2.imread(train_img_path)

# print(f"third image:{train_img_path}")
# result = stitchPairOfImages(query_img, train_img)