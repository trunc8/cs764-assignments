#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


read_dir = os.path.join(sys.path[0], "../data/")
save_dir = os.path.join(sys.path[0], "../results/")

# Parse arguments for "-mat"
parser = argparse.ArgumentParser(description='Method of obtaining Transform matrix')
parser.add_argument("-mat", default="api")
args = parser.parse_args()
mat = args.mat

distorted = cv2.imread(read_dir + "distorted.jpg")

# Shape = (rows, cols, channels)
desired_rows, desired_cols = 600, 600

# Correspondence points: [x,y] Obtained manually from cv2.imshow
pts1 = np.float32([[0, 0],
                   [601, 61],
                   [61, 601]])
pts2 = np.float32([[0, 0],
                   [600, 0],
                   [0, 600]])

if mat == "api":
  H = cv2.getAffineTransform(pts1, pts2)

elif mat == "manual":
  ## TODO: Manually derive 2*3 transformation matrix M
  '''
  H*X = Y
  H: 2*3 Direct Linear Transform matrix
  X: 3*3 Input image pixels in homogeneous coordinates
  Y: 2*3 Output image pixels in euclidean coordinates
  '''
  X = pts1.transpose()
  I = np.ones((1,3))
  X = np.row_stack((X, I))
  Y = pts2.transpose()
  H = Y@np.linalg.inv(X)

else:
  print("Incorrect input for method. Aborting!")
  sys.exit(0)

original = cv2.warpAffine(distorted, H, (desired_cols, desired_rows)) 

cv2.imshow("Distorted Chessboard", distorted)
cv2.imshow("Original Chessboard", original)

cv2.waitKey(0)
cv2.destroyAllWindows()