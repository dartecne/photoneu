import os
import pandas as pd
import cv2 as cv
import argparse
import numpy as np
import time
from matplotlib import pyplot as plt

ruta_carpeta = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\labeled-data-ordered'
#ruta_carpeta_2 = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\trimice-dlc-2021-06-22\labeled-data\videocompressed1_labeled'
#ruta_carpeta_2 =r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\labeled-data-ordered'
ruta_carpeta_2 =r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\trimice-dlc-2021-06-22\evaluation-results\iteration-0\trimiceJun22-trainset70shuffle1\LabeledImages_DLC_mobnet_35_trimiceJun22shuffle1_10000_snapshot-10000'
#ruta_imagen = r'\Training-videocompressed1-img1816.png'
ruta_imagen = r'\Training-videocompressed4-img1075.png'
#video_path = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\videos_ratoncillos\markerless_1.MP4'
video_path =r'C:\Users\inges\OneDrive - UDIT\src\photoneu\videos_ratoncillos\Video_5.MP4'
#ruta_imagen = r'\img0000_individual.png'


##############################################

#kernel_5 = np.ones((5,5),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.
#kernel_3 = np.ones((3,3),np.uint8) #Define a 3×3 convolution kernel with element values of all 1.
kernel_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
kernel_5 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
#cap = cv.VideoCapture(args.camera)
#img_path = r'/home/ratoncillos/OneDrive/src/photoneu/dataset/deeplabcut/labeled-data-ordered/img0253.png'

RESIZE_FACTOR = 2.0
MAJOR_DEFECT_THRESHOLD = 12.0 / RESIZE_FACTOR #6.0 5.0 12.0
MIN_AREA = 800
MAX_AREA = 12 * MIN_AREA#2.5 * MIN_AREA
thres = 49 # min B/W value for threshold
N = 113 # number of images
#N = 242 # frames video_2 video_4
#N = 302 # framse video_3
#N = 1801 # framse video_5
#N = 1801 # framse video_6


normal_size = (480, 640)
x_crop_min = int(normal_size[1]/10) #50
x_crop_max = int(normal_size[1]/20) # 30
y_crop_min = int(normal_size[0]/16) # 30
y_crop_max = int(normal_size[0]/20) # 16 -- 40

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
    contour = contour.squeeze()  # Elimina dimensiones extra si es necesario
    contour.tolist()
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
          start = tuple(contour[s])
          end = tuple(contour[e])
          far = tuple(contour[f])
          real_far_dist = d / 256.0
          if real_far_dist >= MAJOR_DEFECT_THRESHOLD:
               intersections.append(f)
               inter_points.append(far)
    n_points = len(intersections)
    if (n_points == 0):
        split_contours = [contour]
    elif n_points == 1:
        print("intersections[0]:", intersections[0])
        print("len(contour):", len(contour))
        print("len(contour)//2:", len(contour)//2)
        index_plus = intersections[0] + len(contour)//2
        index_less = intersections[0] - len(contour)//2
        if intersections[0] < len(contour)//2:
            blob["segments"] = [
                contour[intersections[0]:index_plus]
                , np.vstack([contour[index_plus:],contour[:intersections[0]+1]])
    #            ,contour[midle_index:] + contour[:intersections[0]]
            ]
        else:
            blob["segments"] = [
                np.vstack([contour[intersections[0]:],contour[:index_less]])
                ,contour[index_less:intersections[0]+1]
                ]
        split_contours = [
            blob["segments"][0], blob["segments"][1]
        ]
    elif n_points == 2:
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , np.vstack([contour[intersections[1]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1]
        ]
    elif n_points == 3:
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , contour[intersections[1]:intersections[2]+1]
            , np.vstack([contour[intersections[2]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1], blob["segments"][2]
        ]
    elif n_points == 4:
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
       for i in range(0,len(contour),n_points):
          split_contour = contour[i:i + n_points]
          if len(split_contour)> 5:
             split_contours.append(split_contour)
    blob["split_contours"] = split_contours
    for c in split_contours:
        if len(c) >= 5:
            blob["ellipses"].append(cv.fitEllipse(c))            
        else:
            blob["ellipses"].append(None)            
    blob["intersections"] = intersections    
    blob["inter_points"] = inter_points    
    return blob["ellipses"]
     
def detectMice( frame, high_V, t_init_resize ): 
     SEGMENT_COLORS = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]

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
     opening = cv.morphologyEx(frame_erode, cv.MORPH_OPEN, kernel_3, iterations = 6)
     e7 = cv.getTickCount()
    
     blobs = detect_blobs( opening )
     nc = len(blobs)
#     print("Found %2d blob(s)." % len(blobs))
     if len(blobs) > 0: 
        for blob in blobs:
            blob["areas"] = []
            blob["obb"] = []
            blob["mice"] = []
            e = process_blob(blob)
            for n, split_contour in enumerate(blob["split_contours"]):
#                area = cv.contourArea( split_contour )
#                blob["areas"].append( area )
                rect = cv.minAreaRect(split_contour) # center, size, angle
                blob["obb"].append( rect )
                area = rect[1][0] * rect[1][1]
                blob["areas"].append( area )
                box = cv.boxPoints(rect)
                box = np.intp(box)       
#                cv.polylines(frame_defects, [split_contour], False, SEGMENT_COLORS[n%4], 2)
                cv.polylines(frame_defects, [split_contour], False, SEGMENT_COLORS[n%4], 2)         
            for n, p in enumerate(blob["inter_points"]):
                cv.circle(frame_defects, p, 3, (0,0,255))
            for n, e in enumerate(blob["ellipses"]):
                if e == None:
                    continue
                area = blob["areas"][n]
                print(area)
                if( area > MIN_AREA) & (area < MAX_AREA ) & (e[0][0] > 0) & (e[0][1] > 0):
                    cv.ellipse(frame, e, (255,30,25), 1)
                    cv.circle(frame,(int(e[0][0]),int(e[0][1])), 3, (255,255,25))
    #                cv.putText(frame,"mice",(int(e[0][0]-40),int(e[0][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
                    cv.putText(frame,str(int(e[0][0]))+","+str(int(e[0][1])),(int(e[0][0]-40),int(e[0][1]+20)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
                    cv.putText(frame,str(round(area)),(int(e[0][0]+10),int(e[0][1]-5)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
    #                cv.drawContours(frame,[box],0,(255,0,25),2)
                    blob["mice"].append(e)
     e8 = cv.getTickCount()
     cv.imshow("opening", opening)
     cv.imshow("Detection of defects", frame_defects)


     t_gray_norm = (e3 - e2)/cv.getTickFrequency()
     t_blur = (e4 - e3)/cv.getTickFrequency()
     t_thres = (e5 - e4)/cv.getTickFrequency()
     t_erosion = (e6 - e5)/cv.getTickFrequency()
     t_opening = (e7 - e6)/cv.getTickFrequency()
     t_blob = (e8 - e7)/cv.getTickFrequency()

     latencies = [t_init_resize, t_gray_norm,t_blur,t_thres,t_erosion,t_opening, t_blob]
     return latencies, blobs, frame, frame_defects    

# Función para leer los nombres de las imágenes de una carpeta
def leer_imagenes_de_carpeta(carpeta):
    # Lista para almacenar los nombres de los archivos
    nombres_imagenes = []
    
    # Recorrer todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        # Comprobar si el archivo es una imagen (puedes agregar más extensiones si es necesario)
        if archivo.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            archivo = "/"+archivo
            nombres_imagenes.append(archivo)
    
    # Crear un DataFrame de Pandas con los nombres de las imágenes
    df = pd.DataFrame(nombres_imagenes, columns=['img_path'])
    
    return df

def analyze_image_from_path(i, ruta_carpeta, img_name):  
#    img_base = os.path.basename(img_name)
    img_base = os.path.splitext(img_name)[0]
    img_in = ruta_carpeta + img_name
    e1 = cv.getTickCount()
    frame = cv.imread(img_in)
    frame_o = frame[x_crop_min:(normal_size[0]-x_crop_max), y_crop_min:(normal_size[1]-y_crop_max)]
    frame = cv.resize(frame_o, (int(normal_size[1]/RESIZE_FACTOR), int(normal_size[0]/RESIZE_FACTOR))) # frame.shape/2
#     print(frame.shape) # 240 x 320
    e2 = cv.getTickCount()
    t_init_resize = (e2 - e1)/cv.getTickFrequency()

    _,_, final_frame,defects_frame = analyze_image(i, frame = frame, t_init_resize = t_init_resize)
    write_image(ruta_carpeta, img_name, final_frame)
    write_image(ruta_carpeta, img_base + "_c" + ".png", defects_frame)


def analyze_image( i, frame, t_init_resize ):  
    t, b, final_frame, convex_frame = detectMice(frame, thres, t_init_resize)
    k = 0
    nip = 0
    for j, blob in enumerate(b):
        kk = 0
        for jj, m in enumerate(blob["mice"]):
            cx, cy = m[0]
            if k < 3: 
                mus_x[k,i] = int(cx)
                mus_y[k,i] = int(cy)
                a = int(blob["areas"][kk])
                area[k,i] = a
                k += 1
                kk += 1
        nip += len(blob["intersections"])
    times.append(t)
    nb.append(len(b))
    num_inter_points.append(nip)
    if nip != 0:
        if_overlaps.append(1)
    else:
        if_overlaps.append(0)
    num_mice_per_blob = 0
    for n, blob in enumerate(b):
        num_mice_per_blob += len(blob["mice"])
    nm.append(num_mice_per_blob)
    return t, b, final_frame, convex_frame
    
def write_image(ruta_carpeta, img_name, final_frame):
    img_out = ruta_carpeta + "/no_labels/"+ str(thres) +"/"
    try:
        os.mkdir(img_out)
    except Exception as e:
        print(f"{e}")
    img_out += img_name  
    print(img_out)
    cv.imwrite(img_out, final_frame)
    
def analyze_video(video_path):
    cap = cv.VideoCapture(video_path)
#    frame_width = int(cap.get(3))
#    frame_height = int(cap.get(4))
    frame_width = int(normal_size[0]/RESIZE_FACTOR)
    frame_height = int(normal_size[1]/RESIZE_FACTOR)
    convex_out = cv.VideoWriter('no_labels_convex_5.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_height,frame_width)) 
    video_out = cv.VideoWriter('no_labels_5.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_height,frame_width))
    i = 0

    while cap.isOpened():
        e1 = cv.getTickCount()
        ret, frame = cap.read()
        if frame is None:
            print("frame ERROR")
            break
        frame = cv.resize(frame, (frame_height, frame_width)) # frame.shape/2
        e2 = cv.getTickCount()
        t_init_resize = (e2 - e1)/cv.getTickFrequency()
        _,_,frame_out, frame_convex = analyze_image(i, frame, t_init_resize)
        video_out.write(frame_out)
        convex_out.write(frame_convex)
        cv.imshow("video", frame_out)
        i+=1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    print("n_frames = " + str(i))
    cap.release()
        
def updateDataFrame():
    df_times = pd.DataFrame(times, columns=['t_init_resize', 't_gray_norm','t_blur','t_thres','t_erosion','t_opening', 't_blobs'])
    df_tmp = df.assign(num_mice = nm)
    df_tmp = df_tmp.assign(num_blobs = nb)
    df_tmp = df_tmp.assign(num_inter_points = num_inter_points)
    df_tmp = df_tmp.assign(overlaps = if_overlaps)
    df_tmp = df_tmp.assign(mus_1_x = mus_x[0])
    df_tmp = df_tmp.assign(mus_2_x = mus_x[1])
    df_tmp = df_tmp.assign(mus_3_x = mus_x[2])
    df_tmp = df_tmp.assign(mus_1_y = mus_y[0])
    df_tmp = df_tmp.assign(mus_2_y = mus_y[1])
    df_tmp = df_tmp.assign(mus_3_y = mus_y[2])
    df_tmp = df_tmp.assign(mus_1_area = area[0])
    df_tmp = df_tmp.assign(mus_2_area = area[1])
    df_tmp = df_tmp.assign(mus_3_area = area[2])
#    print(df_times)
    print(df_tmp[df_tmp["num_mice"] >= 3])
    print("n_total_mice = " + str(sum( df_tmp["num_mice"] )))
    print("n_files_3_mices = " + str(df_tmp[df_tmp["num_mice"] == 3].shape[0]))
    fn = "no_labels_test_" + str(thres) + ".csv"
    df_times.to_csv('no_labels_test_times.csv')
    print(fn)
    df_tmp.to_csv(fn)

### MAIN #################
df = leer_imagenes_de_carpeta(ruta_carpeta)
df_tmp = pd.DataFrame()
df_times = pd.DataFrame()
nb = []
nm = []
if_overlaps = []
num_inter_points = []
times = []
coord = [None]*3
mus_x = np.full((3,N), None)
mus_y = np.full((3,N), None)
area = np.full((3,N), None)

img = ruta_carpeta + df['img_path'][2]

cols = ["mus_1_x", "mus_1_y", "mus_1_area", "mus_2_x", "mus_2_y", "mus_2_area", "mus_3_x", "mus_3_y", "mus_3_area"]

# Para detectar mice en todas las imagenes del directorio
for i, img in enumerate(df['img_path']):
    analyze_image_from_path(i, ruta_carpeta, img)
updateDataFrame()

#analyze_image_from_path(0,ruta_carpeta = ruta_carpeta_2, img_name = ruta_imagen)
#analyze_video( video_path )
#cv.destroyAllWindows()
