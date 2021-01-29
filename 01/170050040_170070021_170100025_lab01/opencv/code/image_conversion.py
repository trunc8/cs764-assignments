#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import sys

bgr_img = cv2.imread(sys.argv[1])
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
normalized_bgr = bgr_img/255.
normalized_rgb = rgb_img/255.

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title("Original image")

f.add_subplot(1, 2, 2)
plt.title("Normalized image")
plt.imshow(normalized_rgb)
plt.show()


cv2.imshow("Original image", bgr_img)
cv2.imshow("Normalized image", normalized_rgb)
cv2.waitKey(0)