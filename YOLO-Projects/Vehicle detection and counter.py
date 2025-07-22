from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture('highway.mp4')

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
              "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis","snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
              "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
              "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask=cv2.imread('mask.png')

# Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits=[500,750,700,750]

while True:
    success, img = cap.read()

    imgRegion=cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)

    detection=np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            # This line is used to get and round off the confidence score (how sure YOLO is about the object it detected) to 2 decimal places
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])  # Get class ID
            currentClass=classNames[cls]
            # This line is used to print the conf and class name
            if currentClass=="car" or currentClass=='truck' or currentClass=='bus' or currentClass=='mortorbike' and conf>0.5:
               cvzone.putTextRect(img, f'{currentClass}{conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1)
               cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
               currentArray=np.array([x1,y1,x2,y2,conf])
               detection=np.vstack((detection,currentArray))

    resultsTracker=tracker.update(detection)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,Id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,offset=10)

    scaled_img = cv2.resize(img, (960, 540))
    scaled_region = cv2.resize(imgRegion, (960, 540))
    cv2.imshow('frame', scaled_img)
    cv2.imshow("imageRegion", scaled_region)

    # Wait for 1 ms and check if 'q' is pressed to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



