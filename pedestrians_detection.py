import cv2 as cv
import numpy as np 
from imutils.object_detection import non_max_suppression # Handle overlapping

filename = 'img/pedestrians.mp4'

def pedestrians_detection():
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector()) # Initialize the People Detector
    
    cap = cv.VideoCapture(filename) # Load a video

    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            orig_frame = frame.copy()
            (bounding_boxes, _) = hog.detectMultiScale(frame, 
                                                       winStride=(4, 4), # winStride: step size in x and y direction of the sliding window
                                                       padding=(4, 4), # padding: no. of pixels in x and y direction for padding of sliding window
                                                       scale=1.09) # scale: Detection window size increase coefficient

            # Draw bounding boxes on the frame
            for (x, y, w, h) in bounding_boxes: 
                    cv.rectangle(orig_frame, 
                    (x, y),  
                    (x + w, y + h),  
                    (0, 0, 255), 
                    2)
            
            # Get rid of overlapping bounding boxes
            bounding_boxes = np.array([[x, y, x + w, y + h] for (
                                        x, y, w, h) in bounding_boxes])
                    
            selection = non_max_suppression(bounding_boxes, 
                                            probs=None, 
                                            overlapThresh=0.45)

            # Draw the final bounding boxes
            for (x1, y1, x2, y2) in selection:
                cv.rectangle(frame, 
                            (x1, y1), 
                            (x2, y2), 
                            (0, 255, 0), 
                            4)

            
           
            cv.imshow("Pedestrians Detection", frame)  

            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()