import glob
import shutil
import random
import cv2
from PIL import Image
import numpy as np


def apply_filters(original):
    bwimage = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    bwimage = cv2.fastNlMeansDenoising(bwimage, None,15,7,21)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    bwimage = clahe.apply(bwimage)

    bwimage = cv2.GaussianBlur(bwimage, (5,5), 0)
    bwimage = cv2.adaptiveThreshold(bwimage,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,17,5)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bwimage = cv2.morphologyEx(bwimage, cv2.MORPH_CLOSE, kernel)
    return bwimage


def resizeImage (img):
    '''
    Resize img 28x28
    '''
    
    target_size = 28 #width and border = 28
    border_width = 0
    
    old_size = img.shape[:2]
    ratio = float(target_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    img = cv2.resize(img,(new_size[1],new_size[0]), 0, 0, interpolation = cv2.INTER_AREA)
  
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    new_img = cv2.copyMakeBorder(new_img, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    return new_img

def prepare_dataset():
    for i in range(0, 10):
        path = f'data/{i}'
        images = [f for f in glob.glob(path + "**/*.png")]
        print(len(images))
        #break
        num_images_train = int(len(images) * 0.98)
        train_images = random.sample(images, num_images_train)
        test_images = list(set(images) - set(train_images))
        dimension = (28, 28)
    
        for train_image_path in train_images:
            original = cv2.imread(train_image_path)
            cv2.imwrite(f'data/train/{train_image_path[-18:]}', resizeImage(apply_filters(np.array(original))))
    
    
        for test_image_path in test_images:
            original = cv2.imread(test_image_path)
            cv2.imwrite(f'data/test/{test_image_path[-18:]}', resizeImage(apply_filters(np.array(original))))

