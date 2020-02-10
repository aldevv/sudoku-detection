
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt


def apply_filters(original):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # para eliminar ruido en la imagen segun la documentacion, para preparar a la deteccion de bordes
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # para aislar las lineas de lo demas
    threshold = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    # nos interesan las lineas NEGRAS, esto hace que en la imagen sean visibles como lineas blancas
    inverted_threshold = cv2.bitwise_not(threshold)
    # el kernel es un tipo de slider que recorrera toda la imagen, es usado por la funcion dilate
    kernel = np.ones((3, 3), np.uint8)
    # usado para arreglar las lineas que hayan quedado desconectadas
    dilatation = cv2.dilate(inverted_threshold, kernel)
    return dilatation


def find_big_square(dilatation):
    contours, hierarchy = cv2.findContours(
        dilatation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print(f"num contours: {len(contours)}")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[
        :10]  # areas mas grandes primero
    big_square = None
    maxPerimeter = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)  # largo del contour
        approx = cv2.approxPolyDP(
            contour, 0.02*perimeter, True)  # constante 0.02
        if len(approx) == 4:
            if perimeter > maxPerimeter:  # para conseguir el contorno mas grande
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

    top_left_sum = 9999999
    bottom_right_sum = -1
    # encuentra los punto esquina arriba izquirda y esquina abajo derecha
    for i in big_square:
        add = i[0][0] + i[0][1]
        if(add < top_left_sum):
            top_left_sum = add
            top_left_point = tuple(i[0])
        if(add > bottom_right_sum):
            bottom_right_sum = add
            bottom_right_point = tuple(i[0])

    non_calculated_points = []
    # ahora reunimos los puntos los cuales no conocemos su ubicacion
    for i in big_square:
        if(i[0][0] not in top_left_point and i[0][0] not in bottom_right_point):
            non_calculated_points.append(list(i[0]))

    # si la x es mayor en alguna de ellas implica que sera en la derecha, dado que el unico punto que queda en la derecha es esquina arriba derecha
    # entonces ese es el punto, de lo contrario es el otro
    if(non_calculated_points[0][0] > non_calculated_points[1][0]):
        top_right_point = tuple(non_calculated_points[0])
        bottom_left_point = tuple(non_calculated_points[1])
    else:
        top_right_point = tuple(non_calculated_points[1])
        bottom_left_point = tuple(non_calculated_points[0])

    print("top left: ", top_left_point)
    print("top right", top_right_point)
    print("bottom left", bottom_left_point)
    print("bottom right", bottom_right_point)

    # sabemos que la esquina superior izquierda es min_x y min_y, la esquina inferior izquierda es min_x y max_y, etc
    # corners = np.float32([(min_x,min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)])
    corners = np.float32([top_left_point, top_right_point,
                          bottom_left_point, bottom_right_point])
    np.array(corners)
    return corners


def draw_corners(original, corners):
    for corner in corners:
        # deben ser tuplas para poder dibujarlas
        cv2.drawMarker(original, tuple(corner), (0, 191, 255), 0, 20, 3)


def transform(original, corners):
    x = 450
    y = 450
    # puntos de las esquinas de la nueva imagen
    new_size = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
    M = cv2.getPerspectiveTransform(corners, new_size)
    size = np.float32([x, y])  # dimensiones nueva imagen
    result = cv2.warpPerspective(original, M, tuple(size))
    return result


def find_all_squares(img):  # infer 81 cells from image
    squares = []
    side = img.shape[:1]
    side = side[0] / 9

    for i in range(9):  # get each box and append it to squares -- 9 rows, 9 cols
        for j in range(9):
            p1 = (i*side, j*side)  # top left corner of box
            p2 = ((i+1)*side, (j+1)*side)  # bottom right corner of box
            squares.append((p1, p2))
    return squares


def draw_squares_to_image(in_img, rects, colour=255):

    img = in_img.copy()
    for rect in rects:
        img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(
            int(x) for x in rect[1]), colour)
    return img


def get_numbers(image, squares):
    x = [[int(a[0][0]), int(a[1][0])] for a in squares]
    y = [[int(b[0][1]), int(b[1][1])] for b in squares]
    # cv2.imshow('originalT', originalT[0:50,0:50])
    number_images = []
    for row, col in zip(x, y):
        number_images.append(image[row[0]:row[1], col[0]:col[1]])
    return number_images

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


# cv2.imwrite('houghlines5.jpg',gray)

# for canny and hough
# edges = cv2.Canny(filteredT,50,150, apertureSize = 3) # experimentar con canny

# minLineLength = 100
# maxLineGap = 10
# linesP = cv2.HoughLinesP(filteredT,1,np.pi/180,50, None,minLineLength,maxLineGap)


# if linesP is not None:
# for i in range(0, len(linesP)):
# l = linesP[i][0]
# cv2.line(originalT, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


# plt.imshow(transformed)
# plt.show()
