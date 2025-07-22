from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
results = model('C:/Users/verma/AIML/OPENCV/YOLO/Images/2.png',show=True)

# # Show the image manually using OpenCV
# result_img = results[0].plot()  # Draw boxes and labels on image
# cv2.imshow("Detection Result", result_img)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
