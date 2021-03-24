#!/usr/bin/env python


import argparse
import cv2
import sys, os


parser = argparse.ArgumentParser()
parser.add_argument('-w', default=1.5)
parser.add_argument('-l', default=5)
parser.add_argument('-b', default=3)
parser.add_argument('-x', default=60)
parser.add_argument('-y', default=40)
parser.add_argument('-theta', default=45)

args = parser.parse_args()

w = int(args.w)
l = int(args.l)
b = int(args.b)
x = int(args.x)
y = int(args.y)
theta = int(args.theta)

def get_data():
  directory = os.path.join(sys.path[0], "../data/augment_book/")
  images = [f for f in os.listdir(directory) if 
          os.path.isfile(os.path.join(directory,f))]
  images.sort()
  NUMBER_OF_WALL_IMAGES = 8
  front_path = os.path.join(directory, images[NUMBER_OF_WALL_IMAGES])
  front = cv2.imread(front_path)
  side_path = os.path.join(directory, images[NUMBER_OF_WALL_IMAGES+2])
  side = cv2.imread(side_path)
  wall_path = os.path.join(directory, images[0])
  wall = cv2.imread(wall_path)

  # cv2.imshow("front", front)
  # cv2.imshow("side", side)
  # cv2.waitKey(0)

get_data()