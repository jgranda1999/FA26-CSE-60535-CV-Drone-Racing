# -*- coding: utf-8 -*-

# Graduate Computer Vision (CSE 60535)
# University of Notre Dame, Fall 2026
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp; first version: September 2017

# Here are your tasks:
#
# Task 1:
# - Select one object that you want to track and set the RGB
#   channels to the selected ranges (found by colorRangeExplorer.py).
# - Check if HSV color space works better. Can you ignore one or two
#   channels when working in HSV color space? Why?
# - Try to track candies of different colors (blue, yellow, green).
# 
# Task 2:
#   Adapt your code to distinguihs and track multiple objects of *the same* color 
#   simultaneously, and show them as separate objects in the camera stream.
#
# Task 3:
#   Adapt your code to distinguihs and track multiple objects of *different* colors simultaneously,
#   and show them as separate objects in the camera stream. Make your code elegant 
#   and requiring minimum changes when the number of different objects to be detected increases.

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while (True):
    retval, img = cam.read()

    res_scale = 0.5 # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)


    #######################################################
    # Use colorRangeExplorer.py to find good color ranges for your object(s):

    # Detect selected color (NOTE: OpenCV uses BGR instead of RGB)
    # This example is tuned to blue, in a relatively dark room
    lower = np.array([50, 0, 0])
    upper = np.array([100, 50, 50])
    objmask = cv2.inRange(img, lower, upper)

    # Uncomment this if you want to use HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([176,180,110])
    # upper = np.array([180,210,220])
    # objmask = cv2.inRange(hsv, lower, upper)
    #######################################################

    
    # You may use this for debugging
    cv2.imshow("Binary image", objmask)

    # Resulting binary image may have large number of small objects.
    # You may check different morphological operations to remove these unnecessary
    # elements. You may need to check your ROI defined in step 1 to
    # determine how many pixels your object may have.
    kernel = np.ones((5,5), np.uint8)
    objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
    objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)
    cv2.imshow("Image after morphological operations", objmask)

    # Find connected components
    cc = cv2.connectedComponents(objmask)
    ccimg = cc[1].astype(np.uint8)

    # Find contours of these objects
    contours, hierarchy = cv2.findContours(ccimg,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # We are using [-2:] to select the last two return values from the above function to make the code work with
    # both opencv3 and opencv4. This is because opencv3 provides 3 return values but opencv4 discards the first.

    # You may display the contour points if you want:
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # Ignore bounding boxes smaller than "minObjectSize"
    minObjectSize = 20;
    

    #######################################################
    # TIP: think if the "if" statement
    # can be replaced with a "for" loop
    if contours:
    #######################################################
    
        # Use just the first contour to draw a rectangle
        x, y, w, h = cv2.boundingRect(contours[0])
        
        #######################################################
        # TIP: you want to get bounding boxes
        # of ALL contours (not only the first one)
        #######################################################

        # Do not show very small objects
        if w > minObjectSize or h > minObjectSize:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(img,            # image
            "Here's my object!",        # text
            (x, y-10),                  # start position
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            0.7,                        # size
            (0, 255, 0),                # BGR color
            1,                          # thickness
            cv2.LINE_AA)                # type of line

    cv2.imshow("Live WebCam", img)

    action = cv2.waitKey(1)
    if action & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()


























