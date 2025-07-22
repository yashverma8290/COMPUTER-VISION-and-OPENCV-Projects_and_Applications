import cv2
import numpy as np

img=cv2.imread('image1.jpg')
img=cv2.resize(img,(300,300))

imgHor=np.hstack((img,img,img))
imgVer=np.vstack((img,img,img))

cv2.imshow('image',imgVer)
cv2.waitKey(0)
cv2.destroyAllWindows()
