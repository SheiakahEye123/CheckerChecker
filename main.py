import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import math

rgbimg = cv2.imread("checkers.png")
cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV)

def findEdge(img):
    grayimg = img[:, :, 2]
    scharrx = np.abs(cv2.Scharr(grayimg, cv2.CV_64F, 1, 0))
    scharry = np.abs(cv2.Scharr(grayimg, cv2.CV_64F, 0, 1))

    final = (scharrx + scharry)

    plt.figure(2)

    plt.imshow(final)

    plt.pause(0.000001)

    print(final.max())

    return (final * 255 / final.max()).astype(np.uint8)

def findCheckers(vchann, edgeimg):
    # kernel = (3,3)
    # dilateimg = cv2.dilate(edgeimg, kernel, iterations=2)
    thresh = cv2.adaptiveThreshold(edgeimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -25)
    # erodeimg = cv2.erode(thresh, kernel)
    plt.figure(0)
    plt.imshow(thresh)
    plt.pause(0.001)
    # circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 50, param1=10, param2=2, minRadius=0, maxRadius=15)
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     cv2.circle(vchann, (i[0], i[1]), i[2], 0, 2)

    c, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(c)):
        p2 = math.pow(cv2.arcLength(c[i], True),2)
        a = cv2.contourArea(c[i])
        if a > 20 and 0.9 <= (p2/a)/(4 * 3.14) <= 1.2:
            cv2.drawContours(rgbimg, c, i, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 2)

    plt.figure(1)
    plt.imshow(rgbimg)
    plt.pause(0.1)


edges = findEdge(rgbimg)
findCheckers(rgbimg[:,:,2].astype(np.uint8), edges)
plt.show()
