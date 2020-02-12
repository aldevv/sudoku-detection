
from cv2 import cv2
import numpy as np
import random as rng

def apply_filters(original):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(11,11),0) #para eliminar ruido en la imagen segun la documentacion, para preparar a la deteccion de bordes 
    threshold = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2) # para aislar las lineas de lo demas
    inverted_threshold = cv2.bitwise_not(threshold) # nos interesan las lineas NEGRAS, esto hace que en la imagen sean visibles como lineas blancas
    #el kernel es un tipo de slider que recorrera toda la imagen, es usado por la funcion dilate
    kernel = np.ones((3,3),np.uint8)
    dilatation = cv2.dilate(inverted_threshold,kernel) # usado para arreglar las lineas que hayan quedado desconectadas
    return dilatation

def filter_squares(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0) #para eliminar ruido en la imagen segun la documentacion, para preparar a la deteccion de bordes 
    thresh = 10 # initial threshold
    # canny_output = cv2.Canny(blur,thresh, thresh**2)
    threshold = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2) # para aislar las lineas de lo demas
    inverted_threshold = cv2.bitwise_not(threshold) # nos interesan las lineas NEGRAS, esto hace que en la imagen sean visibles como lineas blancas
    return inverted_threshold

def contours_numbers(inverted_threshold):
    contours, _ = cv2.findContours(inverted_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10] # areas mas grandes primero
    return contours

def find_big_square(dilatation):
    contours, _ = cv2.findContours(dilatation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print(f"num contours: {len(contours)}")
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10] # areas mas grandes primero
    big_square = None
    maxPerimeter = 0

    for contour in contours:
         perimeter = cv2.arcLength(contour,True) # largo del contour
         approx = cv2.approxPolyDP(contour, 0.02*perimeter , True) #constante 0.02 # aproxima a lineas rectas
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

    # """
    def findTopLeft():
        top_left_sum = 9999999
        for i in big_square:
            add = i[0][0] + i[0][1]
            if(add < top_left_sum):
                top_left_sum = add
                top_left_point = tuple(i[0])
        return top_left_point

    def findBottomRight():
        bottom_right_sum = -1
        for i in big_square:
            add = i[0][0] + i[0][1]
            if(add > bottom_right_sum):
                bottom_right_sum = add
                bottom_right_point = tuple(i[0])
        return bottom_right_point

    def find_non_calculated_points():
    # encuentra los punto esquina arriba izquirda y esquina abajo derecha
        non_calculated_points = []
        # ahora reunimos los puntos los cuales no conocemos su ubicacion
        for i in big_square:
            if (list(i[0]) != list(top_left_point) and list(i[0]) != list(bottom_right_point)):
                non_calculated_points.append(list(i[0]))
        return non_calculated_points

    def find_topRight_and_bottomLeft():
        # si la x es mayor en alguna de ellas implica que sera en la derecha, dado que el unico punto que queda en la derecha es esquina arriba derecha
        # entonces ese es el punto, de lo contrario es el otro
        if(non_calculated_points[0][0] > non_calculated_points[1][0]):
            top_right_point = tuple(non_calculated_points[0])
            bottom_left_point = tuple(non_calculated_points[1])
        else:
            top_right_point = tuple(non_calculated_points[1])
            bottom_left_point = tuple(non_calculated_points[0])
        return top_right_point, bottom_left_point

    top_left_point = findTopLeft()
    bottom_right_point = findBottomRight()
    non_calculated_points = find_non_calculated_points() #
    top_right_point, bottom_left_point = find_topRight_and_bottomLeft()

    
    # print("top left: ",top_left_point)
    # print("top right",top_right_point)
    # print("bottom left",bottom_left_point)
    # print("bottom right",bottom_right_point)

    #sabemos que la esquina superior izquierda es min_x y min_y, la esquina inferior izquierda es min_x y max_y, etc
    # corners = np.float32([(min_x,min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)])
    corners = np.float32([top_left_point, top_right_point, bottom_left_point, bottom_right_point])
    np.array(corners)
    return corners

def draw_corners(original, corners):
    for corner in corners:
        cv2.drawMarker(original, tuple(corner), (0,191,255), 0, 20, 3) # deben ser tuplas para poder dibujarlas

def transform(original, corners,x=450,y=450):
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
    

def zoom_number(image, cont):
    inverted_threshold = filter_squares(image)
    contours =  contours_numbers(inverted_threshold)
    largest_area = 0
    x=0
    y=0
    w=0
    h=0
    curve = 0
    for i, c in enumerate(contours):
        xTemp,yTemp,wTemp,hTemp = cv2.boundingRect(c)
        if( cv2.contourArea(c) > largest_area and \
            (hTemp < image.shape[0]*.9 and wTemp < image.shape[1]*.9)\
                and hTemp > 10 and wTemp > 10): #limites definidos, numero no es menor a 10 ni debe
            largest_area = cv2.contourArea(c)  # medir el 90% de la imagen 
            curve = cv2.arcLength(c,True)
            x,y,w,h = cv2.boundingRect(c)
    # print("contourArea for",cont," is: ",curve,"image h:",image.shape[0],"image w:",image.shape[1])
    # print("contour h:",h," and w: ",w)

    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    if(curve != 0):
        top_left_point = (x,y)
        top_right_point = (x+w,y)
        bottom_left_point = (x,y+h)
        bottom_right_point = (x+w,y+h)
        corners = np.float32([top_left_point, top_right_point, bottom_left_point, bottom_right_point])
        result = transform(image,corners,50,50)
    else:
        result = image
    return result

def get_numbers(image, squares):
    x = [[int(a[0][0]),int(a[1][0])] for a in squares]
    y = [[int(b[0][1]),int(b[1][1])] for b in squares]
    # cv2.imshow('originalT', originalT[0:50,0:50])
    cont =0
    allBoxes = []
    for row,col in zip(x,y):
        cont +=1
        mySquare = image[row[0]:row[1],col[0]:col[1]]
        number = zoom_number(mySquare,cont)
        cv2.imshow('Contours', number)
        allBoxes.append(number)
        cv2.waitKey(0)
    return allBoxes

def recognize():
    pass

def process_sudoku_image(image):
    original = image
    filtered = apply_filters(original)
    big_square = find_big_square(filtered)
    corners = find_corners(big_square)
    corners_image = original.copy()
    # dibuja el contorno externo
    cv2.drawContours(corners_image, [big_square], -1, (0, 255, 0), 3)
    draw_corners(corners_image, corners)  # dibuja las esquinas
    originalT = transform(original, corners)
    filteredT = transform(filtered, corners)
    squares = find_all_squares(originalT)
    originalS = draw_squares_to_image(originalT, squares)
    number_images = get_numbers(originalT, squares)
    #print('number_images', number_images)
    return number_images

process_sudoku_image(cv2.imread("../img/Image2.jpg"))