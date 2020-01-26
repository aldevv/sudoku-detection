
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_filters(original):
    gray = cv2.imread('image9.jpg',cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray,(11,11),0) #para eliminar ruido en la imagen segun la documentacion, para preparar a la deteccion de bordes 
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
    """
    ejemplo del contenido de big square (ndarray)

    [[[ 49  35]]

    [[ 46 468]]

    [[553 462]]

    [[554  38]]]

    """
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
    new_size = np.float32([[0, 0], [700, 0], [0, 600], [700, 600]]) # puntos de las esquinas de la nueva imagen
    M = cv2.getPerspectiveTransform(corners, new_size)
    size = np.float32([700,600]) # dimensiones nueva imagen
    result = cv2.warpPerspective(original, M, tuple(size))
    return result

def find_lines():
    pass
def get_numbers():
    pass
def recognize():
    pass

original = cv2.imread('image9.jpg')
filtered = apply_filters(original)
big_square = find_big_square(filtered)
corners = find_corners(big_square)
cv2.drawContours(original , [big_square], -1, (0,255,0), 3)
draw_corners(original,corners)
originalT = transform(original, corners)
filteredT = transform(filtered, corners)

edges = cv2.Canny(filteredT,50,150, apertureSize = 3)
minLineLength = 100
maxLineGap = 10
linesP = cv2.HoughLinesP(filteredT,1,np.pi/180,50, None,minLineLength,maxLineGap)
# lines = cv2.HoughLines(result,1,np.pi/180,200)
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# print(lines)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(originalT, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
# for rho,theta in lines[0]:
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a*rho
    # y0 = b*rho
    # x1 = int(x0 + 1000*(-b))
    # y1 = int(y0 + 1000*(a))
    # x2 = int(x0 - 1000*(-b))
    # y2 = int(y0 - 1000*(a))

    # cv2.line(result,(x1,y1),(x2,y2),(0,0,255),2)

# for x1,y1,x2,y2 in lines[0]:
    # cv2.line(result,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('original', original)
cv2.imshow('originalT', originalT)
cv2.imshow('transformada', filteredT)
# cv2.imshow('filtrada', filtered)
# cv2.imshow('edges', edges)
cv2.waitKey(0)



# plt.imshow(inverted_threshold)
# plt.imshow(transformed)
# plt.show()

# cv2.imwrite('houghlines5.jpg',gray)
