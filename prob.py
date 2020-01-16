
import cv2
import numpy as np
from matplotlib import pyplot as plt
from lib import perspective_transform

original = cv2.imread('image9.jpg')
gray = cv2.imread('image9.jpg',cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray,(11,11),0) #para eliminar ruido en la imagen
threshold = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2) # para aislar las lineas de lo demas
inverted_threshold = cv2.bitwise_not(threshold) # nos interesan las lineas NEGRAS, esto hace que en la imagen sean visibles como lineas blancas
#el kernel es un tipo de slider que recorrera toda la imagen, es usado por la funcion dilate
kernel = np.ones((3,3),np.uint8)
dilatation = cv2.dilate(inverted_threshold,kernel) # usado para arreglar las lineas que hayan quedado desconectadas

contours, _ = cv2.findContours(dilatation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"num contours: {len(contours)}")
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
maxPerimeter = 0

for contour in contours:
     perimeter = cv2.arcLength(contour,True)
     approx = cv2.approxPolyDP(contour, 0.02*perimeter , True) #constante 0.02
     if len(approx) == 4:
         if perimeter > maxPerimeter:
             maxPerimeter = perimeter
             screenCnt = approx

cv2.drawContours(original , [screenCnt], -1, (0,255,0), 3)
cv2.imshow('le', original)
cv2.waitKey(0)

# cnts = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# print(f"number of cnts: {len(cnts)} ")
# for c in cnts:
    # peri = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    # transformed = perspective_transform(original, approx)
    # break
# edges = cv2.Canny(blur,50,150,apertureSize = 3)
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
    # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# plt.imshow(inverted_threshold)
# plt.imshow(transformed)
# plt.show()

# cv2.imwrite('houghlines5.jpg',gray)
