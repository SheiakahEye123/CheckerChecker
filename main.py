import cv2
import numpy as np

rgbimg = cv2.imread("C:/Users/kaide/OneDrive/Desktop/checkers.png")
cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV)

#(0, 10, 100), (40, 255, 255)

#(135, 10, 100), (179, 255, 255)

def checkChecker(ll,lh, hl, hh, img) :
    low = cv2.inRange(img, ll, lh)
    high = cv2.inRange(img, hl, hh)
    kernel = np.ones((3,3),np.uint8)
    erodeimg = cv2.erode(cv2.bitwise_or(low, high), kernel)
    c, hier = cv2.findContours(erodeimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    checkerlist = []

    # zeroimg = np.zeros_like(img)
    # cv2.drawContours(zeroimg, c, -1, (0,0,255), 1)

    for contours in c:
        if cv2.contourArea(contours) > 20:
            (x,y),radius = cv2.minEnclosingCircle(contours)
            center = (int(x),int(y))
            radius = int(radius)
            checkerlist.append((center, radius))
            # cv2.circle(zeroimg,center,radius,(0,255,0),1)

    return checkerlist

redcs = checkChecker((0, 10, 100), (40, 255, 255), (135, 10, 100), (179, 255, 255), rgbimg)
blackcs = checkChecker((0, 0, 0), (179, 40, 40), (0, 0, 0), (179, 40, 40), rgbimg)

for r in redcs:
    cv2.circle(rgbimg, r[0], r[1], (0,255,0), 2)

for b in blackcs:
    cv2.circle(rgbimg, b[0], b[1], (255, 0, 255), 2)



cv2.imshow("Checker", rgbimg)

cv2.waitKey(0)



