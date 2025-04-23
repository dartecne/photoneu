from __future__ import print_function
import cv2 as cv
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import argparse
import numpy as np
import time
from matplotlib import pyplot as plt

max_value = 255
low_V = 0
high_V = 105 # maximo valor bajo el cual se considera color negro
window_capture_name = 'Video Capture'
low_V_name = 'Low V'
high_V_name = 'High V'

#kernel_5 = np.ones((5,5),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.
#kernel_3 = np.ones((3,3),np.uint8) #Define a 3×3 convolution kernel with element values of all 1.
kernel_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
kernel_5 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
#img_path = r'/home/ratoncillos/OneDrive/src/photoneu/dataset/deeplabcut/labeled-data-ordered/img0253.png'
img_path_sufix = "0253"#"411" #"380" #"0002"# "0253"
img_path = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\labeled-data-ordered\img' #0253.png'
img_path += img_path_sufix
img_path += '.png'

RESIZE_FACTOR = 2.0
MAJOR_DEFECT_THRESHOLD = 6.0 / RESIZE_FACTOR #5.0
MIN_AREA = 900
MAX_AREA = 2.5 * MIN_AREA

normal_size = (480, 640)
x_crop_min = int(normal_size[1]/10) #50
x_crop_max = int(normal_size[1]/20) # 30
y_crop_min = int(normal_size[0]/16) # 30
y_crop_max = int(normal_size[0]/16) # 40

###############
def on_low_V_thresh_trackbar(val):
 global low_V
 global high_V
 low_V = val
 low_V = min(high_V-1, low_V)
 cv.setTrackbarPos(low_V_name, window_capture_name, low_V)

def on_high_V_thresh_trackbar(val):
 global low_V
 global high_V
 high_V = val
 high_V = max(high_V, low_V+1)
 cv.setTrackbarPos(high_V_name, window_capture_name, high_V)

def detect_blobs(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blobs = []
    for i, contour in enumerate(contours):
        orig_x, orig_y, width, height = cv.boundingRect(contour)
        roi_image = image[orig_y:orig_y+height,orig_x:orig_x+width]
        blobs.append({
            "i" : i
            , "contour" : contour
            , "origin" : (orig_x, orig_y)
            , "size" : (width, height)
            , "roi_image" : roi_image
        })
    return blobs
 
def process_blob(blob):
    
    contour = blob["contour"]
    blob["hull"] = cv.convexHull(contour)
    blob["ellipses"] = []

    hull_idx = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull_idx)
    intersections = []
    inter_points = []
    split_contours = []
    if defects is not None :
       for i, defect in enumerate(np.squeeze(defects, 1)):#defects.shape[0]):
          s,e,f,d = defect
          start = tuple(contour[s][0])
          end = tuple(contour[e][0])
          far = tuple(contour[f][0])
          real_far_dist = d / 256.0
          if real_far_dist >= MAJOR_DEFECT_THRESHOLD:
               intersections.append(f)
               inter_points.append(far)
               print("real_far_dist = " + str(real_far_dist))
    n_points = len(intersections)
    if (n_points == 0) | (n_points == 1): #TODO: cuando es == 1, dividir el contorno en 2, desde el punto a la mitad del contorno
        print("0/1 point - One ellipse")    # si el área es mayor al doble del mínimo (p.e. 2000). véase fotograma 380
        split_contours = [contour]
    elif n_points == 2:
        print("2 points - 2 ellipses")
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , np.vstack([contour[intersections[1]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1]
        ]
    elif n_points == 3:
        print("3 points - 2 ellipses")
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , contour[intersections[1]:intersections[2]+1]
            , np.vstack([contour[intersections[2]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1], blob["segments"][2]
        ]
    elif n_points == 4:
        print("4 points")
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , contour[intersections[1]:intersections[2]+1]
            , contour[intersections[2]:intersections[3]+1]
            , np.vstack([contour[intersections[3]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            np.vstack([blob["segments"][0], blob["segments"][2]])
            , np.vstack([blob["segments"][1], blob["segments"][3]])
        ]
    else :
       print( str(n_points) + " points")
       for i in range(0,len(contour),n_points):
          split_contour = contour[i:i + n_points]
          if len(split_contour)> 5:
             split_contours.append(split_contour)
    blob["split_contours"] = split_contours       
    for c in split_contours:
        if len(c) >= 5:
            blob["ellipses"].append(cv.fitEllipse(c))            
    blob["intersections"] = intersections    
    blob["inter_points"] = inter_points    
    
    return blob["ellipses"]

cv.namedWindow(window_capture_name)
cv.createTrackbar(low_V_name, window_capture_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_capture_name , high_V, max_value, on_high_V_thresh_trackbar)
     
while True: 
     e1 = cv.getTickCount()
    # ret, frame = cap.read()
     frame = cv.imread(img_path)
     if frame is None:
        print("frame ERROR")
        break
    # print(frame.shape) # (480, 640, 3)
     frame_o = frame[x_crop_min:(normal_size[0]-x_crop_max), y_crop_min:(normal_size[1]-y_crop_max)]
    # frame = cv.resize(frame, (320, 240)) # frame.shape/2
    # frame = cv.resize(frame, (160, 120)) # frame.shape/4
     frame = cv.resize(frame_o, (int(normal_size[1]/RESIZE_FACTOR), int(normal_size[0]/RESIZE_FACTOR))) # frame.shape/2
     frame_defects = frame.copy()
     e2 = cv.getTickCount()
     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     frame_norm = cv.normalize(frame_gray, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX)
     e3 = cv.getTickCount()
     frame_blur = cv.medianBlur(frame_norm, 5)
     e4 = cv.getTickCount()
     ret, frame_threshold = cv.threshold( frame_blur, high_V , 255, cv.THRESH_BINARY_INV ) #+ cv.THRESH_OTSU )
#     frame_threshold = cv.normalize(frame_threshold, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U) #TODO: Esto hace falta?
     e5 = cv.getTickCount()
     frame_erode = cv.morphologyEx(frame_threshold, cv.MORPH_DILATE, kernel_3, iterations = 4)  
#     frame_erode = cv.morphologyEx(frame_threshold, cv.MORPH_ERODE, kernel_3, iterations = 5)  
     e6 = cv.getTickCount()
     opening = cv.morphologyEx(frame_erode, cv.MORPH_OPEN, kernel_3, iterations = 5)
     e7 = cv.getTickCount()
    
     blobs = detect_blobs( opening )
     SEGMENT_COLORS = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]
    
     print("Found %2d blob(s)." % len(blobs))
     if len(blobs) > 0: 
        for blob in blobs:
            blob["areas"] = []
            e = process_blob(blob)
            cv.drawContours(frame_defects, [blob["hull"]],0, (127,127,255), 2)
    #        cv.drawContours(frame_defects, [blob["contour"]], 0, (255,127,127), 1)
            if "inter_points" in blob:
                  for p in blob["inter_points"]:
                     cv.circle(frame_defects, [p[0],p[1]], 5, (0,25,255), -1)               
            for n, split_contour in enumerate(blob["split_contours"]):
                area = cv.contourArea( split_contour )
                blob["areas"].append( area )
                rect = cv.minAreaRect(split_contour) # center, size, angle
                cx, cy = rect[0]
        #       cv.circle(frame, (int(cx),int(cy)),3,(255,255,25))            
                box = cv.boxPoints(rect)
                box = np.intp(box)
                cv.polylines(frame_defects, [split_contour], False, SEGMENT_COLORS[n%4], 2)
                cv.putText(frame_defects,str(int(area)),(int(cx)-30,int(cy)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),1)# Add character description
    
            for n, e in enumerate(blob["ellipses"]):
                area = blob["areas"][n]
                if( area > MIN_AREA) & (area < MAX_AREA ) & (e[0][0] > 0) & (e[0][1] > 0):
                    cv.ellipse(frame, e, (255,30,25), 2)
                    cv.circle(frame,(int(e[0][0]),int(e[0][1])), 3, (255,255,25))
                    cv.putText(frame,"mice",(int(e[0][0]-40),int(e[0][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),2)# Add character description
                    cv.putText(frame,str(int(e[0][0]))+","+str(int(e[0][1])),(int(e[0][0]-40),int(e[0][1]+20)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),2)# Add character description
    #                cv.drawContours(frame,[box],0,(255,0,25),2)
     e8 = cv.getTickCount()
     fps = cv.getTickFrequency() / (e8 - e1) 
     cv.putText(frame,"tau = " + f"{fps:.0f}" + "FPS",(10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,25,0),1)# Add character description
     #print(int(fps))

     cv.imshow("Frame Original", frame_o)    
     cv.imshow("Frame B/N", frame_gray)    
     cv.imshow("Frame norm", frame_norm)    
     cv.imshow("Frame blur", frame_blur)    
     cv.imshow("Frame Threshold: 80", frame_threshold)
     cv.imshow("Frame Erode", frame_erode)
     cv.imshow("Frame Opening (5x5) x 2", opening)
     cv.imshow(window_capture_name, frame)
     cv.imshow("Detection of defects", frame_defects)

     key = cv.waitKey(30)

     if key == ord('q') or key == 27:
        filename_sufix = img_path_sufix + ".png" 
        cv.imwrite("frameOriginal"+filename_sufix, frame_o)
        cv.imwrite("frame"+filename_sufix, frame)
        cv.imwrite("frameThreshold"+filename_sufix, frame_threshold)
        cv.imwrite("frameBlur"+filename_sufix, frame_blur)
        cv.imwrite("frameErode"+filename_sufix, frame_erode)
        cv.imwrite("frameOpening"+filename_sufix, opening)
        cv.imwrite("frameDefects"+filename_sufix, frame_defects)
        cv.imwrite("frameNorm"+filename_sufix, frame_norm)
        print("t_init_resize = " + str((e2 - e1)/cv.getTickFrequency()))
        print("t_gray_norm = " + str((e3 - e2)/cv.getTickFrequency()))
        print("t_blur = " + str((e4 - e3)/cv.getTickFrequency()))
        print("t_thres = " + str((e5 - e4)/cv.getTickFrequency()))
        print("t_erosion = " + str((e6 - e5)/cv.getTickFrequency()))
        print("t_opening = " + str((e7 - e6)/cv.getTickFrequency()))
        print("END")
        time.sleep(1)
        break
    
