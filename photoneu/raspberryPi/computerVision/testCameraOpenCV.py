
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
im_size = (cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(im_size)
fps = cap.get(cv.CAP_PROP_FPS)
print(fps)
gray = []
blur = []
gausianBlur = []
medianBlur = []
while True:
    ret, frame = cap.read()
# if frame is read correctly ret is True
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (7,7))
        gausianBlur =cv.GaussianBlur(gray, (7,7),0)
        medianBlur = cv.medianBlur(gray, 7)
        cv.imshow(window_capture_name, gray)
        cv.imshow("blur 7x7", blur)
        cv.imshow("gaussianBlur 7x7", gausianBlur)
        cv.imshow("medianBlur 7x7", medianBlur)
    else:
        print("Can't receive ret (stream end?)...")
        break
    if frame is None:
        print("Can't receive frame ...")
        break
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        cv.imwrite("original.jpg", gray)
        cv.imwrite("blur.jpg", blur)
        cv.imwrite("gaussianBlur.jpg", gausianBlur)
        cv.imwrite("medianBlur.jpg", medianBlur)
        break

cap.release()
cv.destroyAllWindows()