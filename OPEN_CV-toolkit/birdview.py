import cv2
import numpy as np

img=cv2.imread('image2.png')
print(img)
width,height= 676,800
points1=np.float32([[111,228],[302,195],[158,502],[366,457]])

points2=np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix=cv2.getPerspectiveTransform(points1,points2)
print(matrix)
imageOutput=cv2.warpPerspective(img,matrix,(width,height))
print(imageOutput)
imageOutput=cv2.resize(imageOutput,(400,500))
cv2.imshow('oldImage',img)
cv2.imshow('image',imageOutput)

cv2.waitKey(0)
cv2.destroyAllWindows()
