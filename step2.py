import cv2
import numpy as np
import time

class NoiseReduction():
    # This variable determines if we want to load color range from memory or use the ones defined here. 
    load_from_disk = True

    # If true then load color range from memory
    if load_from_disk:
        penval = np.load('penval.npy')

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    # Creating A 5x5 kernel for morphological operations
    kernel = np.ones((5,5),np.uint8)

    while(1):
    
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip( frame, 1 )

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # If you're reading from memory then load the upper and lower ranges from there
        if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
        # Otherwise define your own custom values for upper and lower range.
        else:             
            lower_range  = np.array([26,80,147])
            upper_range = np.array([81,255,255])
    
        mask = cv2.inRange(hsv, lower_range, upper_range)
    
        # Perform the morphological operations to get rid of the noise.
        # Erosion Eats away the white part while dilation expands it.
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)

        res = cv2.bitwise_and(frame,frame, mask= mask)

        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
        # stack all frames and show it
        stacked = np.hstack((mask_3,frame,res))
        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()