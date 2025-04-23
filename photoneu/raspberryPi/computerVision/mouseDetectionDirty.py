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
high_V = 60 # maximo valor bajo el cual se considera color negro
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'

kernel_5 = np.ones((5,5),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.
kernel_3 = np.ones((3,3),np.uint8) #Define a 3×3 convolution kernel with element values of all 1.

def on_low_V_thresh_trackbar(val):
 global low_V
 global high_V
 low_V = val
 low_V = min(high_V-1, low_V)
 cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
 global low_V
 global high_V
 high_V = val
 high_V = max(high_V, low_V+1)
 cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

###############
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
x_crop_min = 30
x_crop_max = 30
y_crop_min = 30
y_crop_max = 30
#img_path = r'/home/ratoncillos/OneDrive/src/photoneu/dataset/deeplabcut/labeled-data-ordered/img0253.png'
img_path = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\labeled-data-ordered\img0253.png'


def detect_blobs(image):
#    _,img_binary = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blobs = []
    for i, contour in enumerate(contours):
        orig_x, orig_y, width, height = cv.boundingRect(contour)
        roi_image = image[orig_y:orig_y+height,orig_x:orig_x+width]
#        cv.drawContours(image, [contour], 0, (255,0,0), 3)
        blobs.append({
            "i" : i
            , "contour" : contour
            , "origin" : (orig_x, orig_y)
            , "size" : (width, height)
            , "roi_image" : roi_image
        })
    return blobs
 
def process_blob(blob):
    MAJOR_DEFECT_THRESHOLD = 2.0
    
    contour = blob["contour"]
    blob["hull"] = cv.convexHull(contour)
    blob["ellipses"] = []
    if len(contour) < 5:
       return blob["ellipses"]
    
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
    n_points = len(intersections)
    if (n_points == 0) | (n_points == 1):
        print("One ellipse")
        if len(contour)> 5:
           blob["ellipses"] = [cv.fitEllipse(contour)]#
        blob["split_contours"] = contour
    elif n_points == 2:
        print("2 points - 2 ellipses")
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , np.vstack([contour[intersections[1]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1]
        ]
        blob["ellipses"] = [cv.fitEllipse(c) for c in split_contours]
        blob["split_contours"] = split_contours
    elif n_points == 3:
        print("3 points")
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , contour[intersections[1]:intersections[2]+1]
            , np.vstack([contour[intersections[2]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1], blob["segments"][2]
        ]
        blob["ellipses"] = [cv.fitEllipse(c) for c in split_contours]
        blob["split_contours"] = split_contours
    elif len(intersections) == 4:
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
        blob["ellipses"] = [cv.fitEllipse(c) for c in split_contours]
        blob["split_contours"] = split_contours
    else :
       print( str(n_points) + " points")
       for i in range(0,len(contour),n_points):
          split_contour = contour[i:i + n_points]
          if len(split_contour)> 5:
             split_contours.append(split_contour)
       blob["ellipses"] = [cv.fitEllipse(c) for c in split_contours]
       blob["split_contours"] = split_contours
             
    blob["intersections"] = intersections    
    blob["inter_points"] = inter_points    
    
    return blob["ellipses"]
 
def visualize_blob(blob):
    PADDING = 20
    
    orig_x, orig_y = blob["origin"]
    offset = (orig_x - PADDING, orig_y - PADDING)
    
    input_img = cv.copyMakeBorder(blob["roi_image"]
        , PADDING, PADDING, PADDING, PADDING
        , cv.BORDER_CONSTANT, None, 255)

    adjusted_img = cv.add(input_img, 127) - 63
    output_img_ch = cv.cvtColor(adjusted_img, cv.COLOR_GRAY2BGR)
    output_img_seg = output_img_ch.copy()
    output_img_el = output_img_ch.copy()
    
    cv.drawContours(output_img_ch, [blob["hull"] - offset], 0, (127,127,255), 4)
    cv.drawContours(output_img_ch, [blob["contour"] - offset], 0, (255,127,127), 2)
    
    SEGMENT_COLORS = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]
#    if "segments" in blob:
#        for i in range(4):
#            cv.polylines(output_img_seg, [blob["segments"][i] - offset], False, SEGMENT_COLORS[i], 4)
#        for i in range(4):
#            center = (blob["segments"][i] - offset)[0][0]
#            cv.circle(output_img_ch, center, 4, (0,191,255), -1)
#            cv.circle(output_img_seg, center, 4, (0,191,255), -1)
    if "inter_points" in blob:
         for p in blob["inter_points"]:
            print(p)
            offset_point = (p[0] - offset[0], p[1] - offset[1])
            cv.circle(output_img_ch, offset_point, 3, (0,255,191), -1)
            
    
    for ellipse in blob["ellipses"]:
        offset_ellipse = ((ellipse[0][0] - offset[0], ellipse[0][1] - offset[1]), ellipse[1], ellipse[2])
        cv.ellipse(output_img_el, offset_ellipse, (0,0,255), 2)
    
    cv.imshow('', np.hstack([output_img_ch,output_img_seg, output_img_el]))
#    cv.imwrite('output_%d_ch.png' % blob["i"], output_img_ch)
#    cv.imwrite('output_%d_seg.png' % blob["i"], output_img_seg)
#    cv.imwrite('output_%d_el.png' % blob["i"], output_img_el)
    cv.waitKey() 
    
while True: 
#if True: 
 e1 = cv.getTickCount()
# ret, frame = cap.read()
 frame = cv.imread(img_path)
 if frame is None:
    print("frame ERROR")
#    break
# print( frame.shape)
# frame = frame[x_crop_min:frame.shape[0]-x_crop_max, y_crop_min:frame.shape[1]-y_crop_max]
 frame = cv.resize(frame, (320, 240)) # frame.shape/4
# frame = cv.resize(frame, (160, 120)) # frame.shape/4
# frame = cv.pyrMeanShiftFiltering(frame, 21, 51)
 frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 frame_norm = cv.normalize(frame_gray, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX)
 frame_blur = cv.medianBlur(frame_norm, 5)
# frame_blur = cv.medianBlur(frame_blur, 5)
# frame_blur = cv.medianBlur(frame_blur, 5)
# frame_threshold = cv.inRange( frame_GRAY, low_V, high_V )
# frame_threshold = cv.bilateralFilter(frame_threshold, 5, 350, 350)
# ret, frame_threshold = cv.threshold( frame_blur, high_V, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU )
 #hist_eq = cv.equalizeHist(frame_blur)
 ret, frame_threshold = cv.threshold( frame_blur, high_V , 255, cv.THRESH_BINARY_INV ) #+ cv.THRESH_OTSU )
 frame_threshold = cv.normalize(frame_threshold, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
  
 opening = cv.morphologyEx(frame_threshold, cv.MORPH_OPEN, kernel_3, iterations = 5)
 sure_bg = cv.dilate(opening, kernel_3,iterations=3)
 sure_fg = cv.erode(frame_threshold, kernel_3, iterations=5)
 
# frame_edge = cv.Canny(frame_threshold, 10, 400) # detecta los bordes. 
 dist_transform = cv.distanceTransform( sure_fg, cv.DIST_LABEL_PIXEL, 3) # No vemos efecto alguno respecto a la imagen de entrada
# dist_transform = cv.erode(dist_transform, kernel_3, iterations=3)
 ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, cv.THRESH_BINARY)
 sure_fg = np.uint8(sure_fg)

 unknown = cv.subtract(sure_bg, sure_fg)
 
 ret, markers = cv.connectedComponents(sure_fg)
 markers=markers+1
 markers[unknown==255] = 0
 markers = cv.watershed(frame, markers)
 frame[markers==-1] = [255,0,0]
 
# frame_threshold = cv.blur(frame_threshold, (3,3))
# frame_threshold = cv.adaptiveThreshold( frame_threshold, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#        cv.THRESH_BINARY,63,0 )
 frame_erode = cv.erode(frame_threshold, kernel_3, iterations = 2)
# frame_threshold = cv.morphologyEx( frame_threshold, cv.MORPH_CLOSE, kernel_3,iterations= 1 )
 frame_close = cv.morphologyEx( frame_erode, cv.MORPH_CLOSE, kernel_3,iterations= 3 )
 frame_open_1 = cv.morphologyEx( frame_close, cv.MORPH_OPEN, kernel_5,iterations= 2 )
# frame_open_2 = cv.morphologyEx( frame_open_1, cv.MORPH_OPEN, kernel_3,iterations= 5 )
# _tuple = cv.findContours(frame_open_1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
 blobs = detect_blobs(frame_close)
 print("Found %2d blob(s)." % len(blobs))
 for blob in blobs:
    e = process_blob(blob)
      
 if len(blobs) > 0: 
    for blob in blobs:    # Traverse all contours
      cv.drawContours(frame, [blob["hull"]], 0, (127,127,255), 1)
      cv.drawContours(frame, [blob["contour"]], 0, (255,127,127), 1)
      if "inter_points" in blob:
            for p in blob["inter_points"]:
#               print(p)
               cv.circle(frame, [p[0],p[1]], 3, (0,255,191), -1)               
      for n, split_contour in enumerate(blob["split_contours"]):
         area = cv.contourArea( split_contour )
#      if (area > 800) & (area < 1300):
         rect = cv.minAreaRect(split_contour) # center, size, angle
         cx, cy = rect[0]
         cv.circle(frame, (int(cx),int(cy)),3,(255,255,25))            
         box = cv.boxPoints(rect)
         box = np.intp(box)
      for e in blob["ellipses"]:
         cv.ellipse(frame, e, (0,0,255), 1)
         cv.putText(frame,"mice:"+str(int(area)),(int(e[0][0]),int(e[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description
#            cv.drawContours(frame,[box],0,(255,0,25),2)
 e2 = cv.getTickCount()
 fps = cv.getTickFrequency() / (e2 - e1) 
 cv.putText(frame,"tau = " + f"{fps:.0f}" + "FPS",(10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,25,0),1)# Add character description
 print(int(fps))
 cv.imshow(window_capture_name, frame)
# plt.figure()
# plt.imshow(frame)
# plt.show()
 cv.imshow("Dist Transform", dist_transform)
 cv.imshow("Frame Erode (3x3)", frame_threshold)
# cv.imshow("Frame Threshold: 80", frame_threshold)
# cv.imshow("frame blur", frame_blur)
# cv.imshow("Frame Opening (5x5) x 2", frame_open_1)
# cv.imshow("frame open 2", frame_open_2)
# cv.imshow("Frame Closing (3x3) x3", frame_close)

 key = cv.waitKey(30)

 if key == ord('q') or key == 27:
    print("END")
    break
 time.sleep(3)
    
