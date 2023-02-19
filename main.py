import cv2
import numpy as np
from matplotlib import pyplot as plt

rgbimg = cv2.imread("checkers.png")
cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV)



# def checkChecker(ll,lh, hl, hh, img) :
#     low = cv2.inRange(img, ll, lh)
#     high = cv2.inRange(img, hl, hh)
#     kernel = np.ones((3,3),np.uint8)
#     erodeimg = cv2.erode(cv2.bitwise_or(low, high), kernel)
#     c, hier = cv2.findContours(erodeimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     checkerlist = []
#
#     # zeroimg = np.zeros_like(img)
#     # cv2.drawContours(zeroimg, c, -1, (0,0,255), 1)
#
#     for contours in c:
#         if cv2.contourArea(contours) > 20:
#             (x,y),radius = cv2.minEnclosingCircle(contours)
#             center = (int(x),int(y))
#             radius = int(radius)
#             checkerlist.append((center, radius))
#             # cv2.circle(zeroimg,center,radius,(0,255,0),1)
#
#     return checkerlist

def checkChecker(scarimg):
    adaptiveThresholdImg = cv2.adaptiveThreshold(scarimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 301, 2)

    plt.figure(1)

    plt.imshow(adaptiveThresholdImg)

    plt.pause(0.00001)



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

checkChecker(findEdge(rgbimg))

# redcs = checkChecker(rgbimg)
# blackcs = checkChecker(rgbimg)
#
# for r in redcs:
#     cv2.circle(rgbimg, r[0], r[1], (0, 255, 0), 2)
#
# for b in blackcs:
#     cv2.circle(rgbimg, b[0], b[1], (255, 0, 255), 2)

plt.show()
