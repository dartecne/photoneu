
import cv2 as cv
cv.__version__
window_capture_name = 'Video Capture'

#cap = cv.VideoCapture(0,cv.CAP_V4L )
cap = cv.VideoCapture(0,cv.CAP_ANY )
if not cap.isOpened():
    print("Cannot open camera")
    exit()
print("camera ready!")
cv.namedWindow(window_capture_name)
while True:
    ret, frame = cap.read()
# if frame is read correctly ret is True
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow(window_capture_name, gray)
    else:
        print("Can't receive ret (stream end?)...")
        break
    if frame is None:
        print("Can't receive frame ...")
        break
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
     break

cap.release()
cv.destroyAllWindows()