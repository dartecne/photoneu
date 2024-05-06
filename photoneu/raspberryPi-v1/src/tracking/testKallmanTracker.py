import cv2
import numpy as np

# https://pieriantraining.com/kalman-filter-opencv-python-example/

class Tracker():
    """
    This class represents a tracker object that uses OpenCV and Kalman Filters.
    """

    def __init__(self, id, hsv_frame, track_window):
        """
        Initializes the Tracker object.

        Args:
            id (int): Identifier for the tracker.
            hsv_frame (numpy.ndarray): HSV frame.
            track_window (tuple): Tuple containing the initial position of the tracked object (x, y, width, height).
        """

        self.id = id

        self.track_window = track_window
        self.term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

        # Initialize the histogram.
        x, y, w, h = track_window
        roi = hsv_frame[y:y+h, x:x+w]
        roi_hist = cv2.calcHist([roi], [0, 2], None, [15, 16],[0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Create a Kalman filter object with 4 state variables and 2 measurement variables.
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Set the measurement matrix of the Kalman filter.
        # It defines how the state variables are mapped to the measurement variables.
        # In this case, the measurement matrix is a 2x4 matrix that maps the x and y position measurements to the state variables.
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)

        # Set the transition matrix of the Kalman filter.
        # It defines how the state variables evolve over time.
        # In this case, the transition matrix is a 4x4 matrix that represents a simple linear motion model.
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

        # Set the process noise covariance matrix of the Kalman filter.
        # It represents the uncertainty in the process model and affects how the Kalman filter predicts the next state.
        # In this case, the process noise covariance matrix is a diagonal matrix scaled by 0.03.
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03

        cx = x+w/2
        cy = y+h/2
        
        # Set the initial predicted state of the Kalman filter.
        # It is a 4x1 column vector that represents the initial estimate of the tracked object's state.
        # The first two elements are the predicted x and y positions, initialized to the center of the tracked window.
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        
        # Set the corrected state of the Kalman filter.
        # It is a 4x1 column vector that represents the current estimated state of the tracked object.
        # Initially, it is set to the same value as the predicted state.
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    def update( self, frame, hsvframe):
           # Update the Kalman filter with the current measurement.
        measurement = self.kalman.measurementMatrix * self.kalman.statePre
        self.kalman.update(measurement)
        self.kalman.statePost = self.kalman.transitionMatrix * self.kalman.statePre #+ processNoise

        # Predict the next state of the object.
        prediction = self.kalman.predict()

        return prediction

################ MAIN 


# Open the video file.
cap = cv2.VideoCapture(0)

# Create an empty list to store the tracked senators.
senators = []

# Counter to keep track of the number of history frames populated.
num_history_frames_populated = 0

# Start processing each frame of the video.
while True:
    # Read the current frame from the video.
    grabbed, frame = cap.read()

    # If there are no more frames to read, break out of the loop.
    if not grabbed:
        break
    frame = cv2.blur(frame, (5,5))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)     # Convert from BGR to HSV

    mask = cv2.inRange( hsv,( 0,20,60), \
                      (18, 255, 255) ) 
#    if color == 'red':
    mask_2 = cv2.inRange( hsv, ( 160,20,60), \
                         (180,255, 255))
    mask = cv2.bitwise_or( mask, mask_2 )
    kernel = np.ones((6,6),np.uint8) 
    thresh = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel, iterations = 2 )              # Perform an open operation on the image 


    # Find contours in the thresholded image.
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red rectangles around large contours.
    # If there are no senators being tracked yet, create new trackers.
    should_initialize_senators = len(senators) == 0
    id = 0
    boundingRects = []
    for c in contours:
        # Check if the contour area is larger than a threshold.
        if cv2.contourArea(c) > 500:
            # Get the bounding rectangle coordinates.
            boundingRects.append(cv2.boundingRect(c))
            (x, y, w, h) =cv2.boundingRect(c)
            # Draw a rectangle around the contour.
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # If no senators are being tracked yet, create a new tracker for each contour.
            if should_initialize_senators:
                senators.append(Tracker(id, hsv, (x, y, w, h)))
                
        id += 1

    # Update the tracking of each senator.
    for i, senator in enumerate(senators):
#        senator.kalman.update(frame, hsv_frame)   
        prediction = senator.kalman.predict()
        (x, y, w, h) = boundingRects[i]
        filter_state = senator.kalman.correct(np.array([x, y, w, h], np.float32))

    # Display the frame with senators being tracked.
    cv2.imshow('Senators Tracked', frame)

    # Wait for the user to press a key (110ms delay).
    k = cv2.waitKey(110)

    # If the user presses the Escape key (key code 27), exit the loop.
    if k == 27:
        break
