import cv2
import numpy as np

img=cv2.imread('image1.jpg')
img=cv2.resize(img,(300,300))

kernel=np.ones((5,5),np.uint8)

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


imgBlur=cv2.GaussianBlur(img,(5,5),0)
imgCanny=cv2.Canny(img,50,150)
imgDialation=cv2.dilate(imgCanny,kernel,iterations=1)
igErode=cv2.erode(imgDialation,kernel,iterations=1)

cv2.imshow('Gray Image',imgGray)
cv2.imshow('Blur Image',imgBlur)
cv2.imshow('Canny Image',imgCanny)
cv2.imshow('Dialation Image',imgDialation)
cv2.imshow('IgErode Image',igErode)

cv2.waitKey(0)
cv2.destroyAllWindows()