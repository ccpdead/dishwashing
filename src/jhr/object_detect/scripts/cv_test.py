#!/usr/bin/env python3

import cv2
src = cv2.imread("/home/jhr/depth_image2.png")
if src is not None:
    cv2.imshow("depth_image",src)
    cv2.waitKey(0)

