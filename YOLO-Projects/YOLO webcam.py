
from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# cap=cv2.VideoCapture('cars.mp4')



model=YOLO("yolov8n.pt")
while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)


            # This line is used to get and round off the confidence score (how sure YOLO is about the object it detected) to 2 decimal places
            conf=math.ceil((box.conf[0]*100))/100

            #without cvzone
            # text = f'{conf}'
            # cv2.putText(img, text, (x1, max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 255, 0), 2)

            # Class Name
            cls_id = int(box.cls[0])  # Get class ID
            class_name = model.names[cls_id]  # Convert ID to class name

            # This line is used to print the conf and class name
            cvzone.putTextRect(img, f'{conf}{class_name}', (max(0, x1), max(35, y1)))
    cv2.imshow('frame', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
