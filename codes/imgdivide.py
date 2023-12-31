import cv2
import math

def importimage(filepath):

    jpg_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return jpg_img

def divideimg(img, size):
    
    if len(img.shape) != 2:
        raise Exception('Wrong dimension!')
    
    rows, cols = img.shape/size
    rows, cols = map(lambda x: math.floor(x / size), (rows, cols))
    
    for y in range(rows):
        for x in range(cols):
            block = img[y * size: (y + 1) * size, x * size: (x + 1) * size]
    

jpg_file_path = r"C:\Users\82103\Downloads\example.jpg"