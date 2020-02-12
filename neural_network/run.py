import numpy as np
import cv2
from keras.models import load_model
from celery import Celery
from utils import RedisService
import base64

MODEL_FILE = 'models/model-classifier.h5'
app = Celery('scanner', broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/0")
service = RedisService()


def normalizeInput(x):
    x_norm = x / 255
    x_norm = x_norm.reshape(1, 28, 28, 1).astype('float32')

    return x_norm


def resizeImage(img):
    target_size = 28
    border_width = 0

    old_size = img.shape[:2]
    ratio = float(target_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    img = cv2.resize(img, (new_size[1], new_size[0]),
                     0, 0, interpolation=cv2.INTER_AREA)

    new_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    new_img = cv2.copyMakeBorder(new_img, border_width, border_width,
                                 border_width, border_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return new_img


def classify(imgFile):
    img = imgFile
    bwimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bwimage = cv2.fastNlMeansDenoising(bwimage, None, 15, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    bwimage = clahe.apply(bwimage)

    bwimage = cv2.GaussianBlur(bwimage, (5, 5), 0)
    bwimage = cv2.adaptiveThreshold(
        bwimage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bwimage = cv2.morphologyEx(bwimage, cv2.MORPH_CLOSE, kernel)
    model = load_model(MODEL_FILE)
    digit_image = resizeImage(bwimage)
    prep_image = normalizeInput(digit_image)
    probab = model.predict(prep_image)
    cmr_digit = probab.argmax()
    return cmr_digit


@app.task()
def classify_numbers(serialized_number_images):
    numbers = []
    for serialized_number_img in serialized_number_images:
        deserialized = np.loads(base64.b64decode(serialized_number_img))
        print(type(deserialized))
        numbers.append(int(classify(deserialized)))

    return numbers

# print(classify('o.jpeg'))
