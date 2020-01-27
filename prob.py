
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
    x =450
    y = 450
    new_size = np.float32([[0, 0], [x, 0], [0, y], [x, y]]) # puntos de las esquinas de la nueva imagen
    M = cv2.getPerspectiveTransform(corners, new_size)
    size = np.float32([x,y]) # dimensiones nueva imagen
    result = cv2.warpPerspective(original, M, tuple(size))
    return result

def find_all_squares(img): #infer 81 cells from image
    squares = []
    side = img.shape[:1]
    side = side[0] / 9  

    for i in range(9):  #get each box and append it to squares -- 9 rows, 9 cols
    	for j in range(9):
    		p1 = (i*side, j*side) #top left corner of box
    		p2 = ((i+1)*side, (j+1)*side) #bottom right corner of box
    		squares.append((p1, p2))
    return squares

def draw_squares_to_image(in_img, rects, colour=255):

	img = in_img.copy()
	for rect in rects:
		img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
	return img


def get_numbers():
    pass
def recognize():
    pass

original = cv2.imread('image9.jpg')
filtered = apply_filters(original)
big_square = find_big_square(filtered)
corners = find_corners(big_square)
corners_image = original.copy()
cv2.drawContours(corners_image , [big_square], -1, (0,255,0), 3) # dibuja el contorno externo
draw_corners(corners_image,corners) # dibuja las esquinas
originalT = transform(original, corners)
filteredT = transform(filtered, corners)
squares = find_all_squares(originalT)
originalS = draw_squares_to_image(originalT,squares)
tes = squares[0]

x = [[int(a[0][0]),int(a[1][0])] for a in squares]
y = [[int(b[0][1]),int(b[1][1])] for b in squares]
# cv2.imshow('originalT', originalT[0:50,0:50])

# print("squares: ",squares )
print("x: ",x )
# print("y: ",y )
cv2.imshow('subcuadros', originalS)
for row,col in zip(x,y):
    cv2.imshow("indiv",originalT[row[0]:row[1],col[0]:col[1]])
    cv2.waitKey(0)
# int(row[0]):int(col[0]),int(col[0]):int(col[1])]

# edges = cv2.Canny(filteredT,50,150, apertureSize = 3) # experimentar con canny

# minLineLength = 100
# maxLineGap = 10
# linesP = cv2.HoughLinesP(filteredT,1,np.pi/180,50, None,minLineLength,maxLineGap)


# if linesP is not None:
    # for i in range(0, len(linesP)):
        # l = linesP[i][0]
        # cv2.line(originalT, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

# cv2.imshow('original', original)
# cv2.waitKey(0)
# cv2.imshow('filtro', filtered)
# cv2.imshow('contorno y esquinas', corners_image)
# cv2.imshow('transformada', filteredT)
# cv2.imshow('originalT', originalT)
cv2.imshow('subcuadros', originalS)
cv2.waitKey(0)
cv2.destroyAllWindows() #Close all windows



# plt.imshow(transformed)
# plt.show()

# cv2.imwrite('houghlines5.jpg',gray)
