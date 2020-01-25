
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
# from lib1 import perspective_transform
def apply_filters(original):
    gray = cv2.imread('image9.jpg',cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray,(11,11),0) #para eliminar ruido en la imagen
    threshold = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2) # para aislar las lineas de lo demas
    inverted_threshold = cv2.bitwise_not(threshold) # nos interesan las lineas NEGRAS, esto hace que en la imagen sean visibles como lineas blancas
    #el kernel es un tipo de slider que recorrera toda la imagen, es usado por la funcion dilate
    kernel = np.ones((3,3),np.uint8)
    dilatation = cv2.dilate(inverted_threshold,kernel) # usado para arreglar las lineas que hayan quedado desconectadas
    return dilatation

def find_big_square(dilatation):
    contours, hierarchy = cv2.findContours(dilatation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print(f"num contours: {len(contours)}")
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10] # areas mas grandes primero
    big_square = None
    maxPerimeter = 0

    for contour in contours:
         perimeter = cv2.arcLength(contour,True) # largo del contour
         approx = cv2.approxPolyDP(contour, 0.02*perimeter , True) #constante 0.02
         if len(approx) == 4:
             if perimeter > maxPerimeter: # para conseguir el contorno mas grande
                 maxPerimeter = perimeter
                 big_square = approx
    return big_square

def find_corners(big_square):
    print(big_square)
    min_x = min(big_square[:,:,0])
    max_x = max(big_square[:,:,0])
    min_y = min(big_square[:,:,1])
    max_y = max(big_square[:,:,1])

    #sabemos que la esquina superior izquierda es min_x y min_y, la esquina inferior izquierda es min_x y max_y, etc
    # corners = np.float32([(min_x,min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)])
    corners = np.float32([(min_x,min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)])
    np.array(corners)
    return corners

def draw_corners(original, corners):
    for corner in corners:
        cv2.drawMarker(original, tuple(corner), (0,191,255), 0, 20, 3) # deben ser tuplas para poder dibujarlas

def transform(original, corners):
    new_size = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
    M = cv2.getPerspectiveTransform(corners, new_size) # genera una matriz que permitira hacer la transformacion de la ventana, teniendo los puntos como origen
    size = np.float32([500,600])
    result = cv2.warpPerspective(original, M, tuple(size))
    return result


original = cv2.imread('image9.jpg')
filtered = apply_filters(original)
big_square = find_big_square(filtered)
corners = find_corners(big_square)
cv2.drawContours(original , [big_square], -1, (0,255,0), 3)
draw_corners(original,corners)
result = transform(original, corners)

# cv2.imshow('le', original)
cv2.imshow('le2', result)
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
