import cv2
import numpy as np

def track_object(frame):
    # Initialize the Kalman filter.
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.measurementMatrix = np.array([[1, 0], [0, 1]])
    kf.processNoiseCov = np.array([[0.1, 0], [0, 0.1]])
    kf.measurementNoiseCov = np.array([[0.01, 0], [0, 0.01]])

    # Get the current measurement of the object.
    x, y = (frame.shape[1] // 2, frame.shape[0] // 2)
    measurement = np.array([x, y])

    # Update the Kalman filter with the current measurement.
    kf.predict()
    kf.update(measurement)

    # Predict the next state of the object.
    prediction = kf.predict()

    return prediction

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Track the object.
        prediction = track_object(frame)

        # Draw the prediction on the frame.
        cv2.circle(frame, prediction, 5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()