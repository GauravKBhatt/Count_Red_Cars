# Python program to count red cars
# importing cv2 and numpy library 
import cv2 
import numpy as np 

#importing the video 
cap = cv2.VideoCapture('red.mp4')

# Declaring minimum width and height of the rectangle
min_width_rect=40
min_height_rect=40

#count line position 
count_line_position=200

#Initializing the backgroung subtractor
subtractor=cv2.bgsegm.createBackgroundSubtractorMOG()

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# defining the lower and upper values of HSV, 
# this will detect red colour 
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

#getting the center of detector rectangle
def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

#detected car list, allowable error and the count of vehicle
detect=[]
error=1
count=0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break
    resized=cv2.resize(frame,(708,400))
   

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # Create masks for the red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # Combine the masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    # Apply the mask to the original image (optional)
    mask = cv2.bitwise_and(hsv,hsv, mask=red_mask)
    # converting the colored image into grey
    grey = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    #Blurring the image 
    blur=cv2.GaussianBlur(grey,(3,3),5)
    #subtracting the background 
    sub=subtractor.apply(blur)
    #dilating the video 
    dilat=cv2.dilate(sub,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatvid=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatvid=cv2.morphologyEx(dilatvid,cv2.MORPH_CLOSE,kernel)
    #counting using contours
    countour,h=cv2.findContours(dilatvid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Canny Edge Detector
    #Mask = cv2.Canny(Mask,50,150,apertureSize=5,L2gradient=True)

    #Count line
    cv2.line(resized,(0,200),(708,200),(255,127,0),3)

    #fitting the rectangle in detected cars
    for(i,c) in enumerate(countour):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter=(w>=min_width_rect) and (h>min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2)
        center=center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(resized,center,2,(0,0,255),-1)
        #counting the cars
        for(x,y) in detect:
            if y<(count_line_position+error)and y>(count_line_position-error):
                count+=1
                cv2.line(resized,(0,200),(708,200),(0,127,255),3)    
                detect.remove((x,y))
                print("Vehicle Counter:"+str(count))

    cv2.putText(resized,"Red Cars:"+str(count),(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    #Display of the video
    cv2.imshow('Original', resized) 
    cv2.imshow('Masked', dilatvid) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0) 
cv2.destroyAllWindows()