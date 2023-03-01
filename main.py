import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import math

rgbimg = cv2.imread("checkers.png")
cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV)

def findEdge(img):
    final = np.ones(img.shape[0:2], dtype=float)
    for i in range(2):
        grayimg = img[:, :, i]
        scharrx = np.abs(cv2.Scharr(grayimg, cv2.CV_64F, 1, 0))
        scharry = np.abs(cv2.Scharr(grayimg, cv2.CV_64F, 0, 1))
        final += (scharrx + scharry)

    plt.figure(2)

    plt.imshow(final)

    plt.pause(0.000001)

    return ((1/final) * 255).astype(np.uint8)

def findCheckers(vchann, edgeimg):
    thresh = cv2.adaptiveThreshold(edgeimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 0)
    blur = cv2.blur(thresh,(3,3))
    plt.figure(0)
    plt.imshow(blur)
    plt.pause(0.001)

    c, hier = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(c)):
        p2 = math.pow(cv2.arcLength(c[i], True),2)
        a = cv2.contourArea(c[i])
        if a > 20 and 0.9 <= (p2/a)/(4 * 3.14) <= 1.3:
            cv2.drawContours(rgbimg, c, i, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 2)
        else:
            cv2.drawContours(rgbimg, c, i, (255,0,0), 1)

    plt.figure(1)
    plt.imshow(rgbimg)
    plt.pause(0.1)




edges = findEdge(rgbimg)
findCheckers(rgbimg[:,:,2].astype(np.uint8), edges)
plt.show()
