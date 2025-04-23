import cv2
import numpy as np

def track_object(kf, target):

    # Update the Kalman filter with the current measurement.
    prediction = kf.predict()
    if target != None:
        (x, y, w, h ) = target
        measurement = np.array([x, y], dtype=np.float32)
        kf.correct( measurement )

    # Predict the next state of the object.
        prediction = kf.predict()

    return prediction

if __name__ == "__main__":
    # Initialize the Kalman filter.
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.measurementMatrix = np.array([[1, 0,0,0], [0, 1,0,0]], dtype = np.float32)
    pnc = 1e1 # cuanto mas bajo, mas suave y mas lento es el filtro, aparentemente
    mnc = 1e-3
    ecp = 1e-1
    kf.processNoiseCov = pnc * np.array([[1.,0,0,0], 
                                   [0,1.,0,0], 
                                   [0,0,1.,0], 
                                   [0,0,0,1.]],  dtype=np.float32)
    kf.measurementNoiseCov = mnc * np.array([[1., 0], 
                                       [0, 1.]], dtype=np.float32)
    kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
    
    kf.errorCovPost = ecp * np.array([[1.,0,0,0],
                                [0,1.,0,0],
                                [0,0,1.,0],
                                [0,0,0,1.]], dtype=np.float32)
#    kf.measurementMatrix = np.array([[1, 0], [0, 1]], dtype=np.float32)
#    kf.processNoiseCov = np.array([[0.1, 0], [0, 0.1]], dtype=np.float32)
#    kf.measurementNoiseCov = np.array([[0.01, 0], [0, 0.01]], dtype=np.float32)

    cap = cv2.VideoCapture(0)
    prediction = np.array([0,0])

    while True:
      ret, frame = cap.read()
      if not ret:
          break
      frame = cv2.blur(frame, (5,5))
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)     # Convert from BGR to HSV      mask = cv2.inRange( hsv,( 0,120,60), \
      mask = cv2.inRange( hsv,( 0,120,60), \
                    (18, 255, 255) ) 
      mask_2 = cv2.inRange( hsv, ( 160,20,60), \
                           (180,255, 255))
  #    mask = cv2.bitwise_or( mask, mask_2 )
      kernel = np.ones((6,6),np.uint8) 
      thresh = cv2.morphologyEx( mask_2, cv2.MORPH_OPEN, kernel, iterations = 2 )              # Perform an open operation on the image       senator = Tracker(id, hsv, (200, 300, 20, 20))
      target = None
      (x, y, w, h) = (0,0,0,0)
      # Find contours in the thresholded image.
      contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for c in contours:
      # Check if the contour area is larger than a threshold.
          area = cv2.contourArea(c)
  #        print(area)
          if (area > 700) & (area < 760) :
          # Get the bounding rectangle coordinates.
              target = cv2.boundingRect(c)#        
              (x, y, w, h) = target
          # Draw a rectangle around the contour.
  #            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
              (c_x,c_y), radius = cv2.minEnclosingCircle(c)
              center = (int(c_x), int(c_y))
              radius = int(radius)
              cv2.circle(frame, center, radius,(0,255,100),1)
              cv2.putText(frame,str(x) + "," + str(y),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
              break          # Track the object.
      
      prediction = track_object(kf, target)
      print("prediction:" + str(prediction) + "  target: " + str(x) + "," + str(y) )
      center = (int(prediction[0]), int(prediction[1]))
      # Draw the prediction on the frame.
      cv2.circle(frame, center, 5, (0, 255, 0), 2)          
      cv2.imshow("Frame", frame)
      k = cv2.waitKey(1)           
      if k == 27:
        break      
    cap.release()
    cv2.destroyAllWindows()