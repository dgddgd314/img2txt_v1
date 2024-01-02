import cv2
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.signal import convolve2d

# importing image?""
def importimage(filepath):

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img

# changes single letter into array
def create_image_array_from_text(text, size=(100, 100)):
    # image size and bgcolor
    width, height = size
    background_color = 0

    # make new img file
    image = Image.new("L", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # font
    font_size = size[0]
    font = ImageFont.truetype("arial.ttf", font_size) 

    text_bbox = draw.textbbox((0, 0), text, font=font)

    # drawing in the middle
    x = (width - (text_bbox[2] - text_bbox[0])) // 2 - text_bbox[0]
    y = (height - (text_bbox[3] - text_bbox[1])) // 2 - text_bbox[1]
    draw.text((x, y), text, fill=255, font=font) 
    
    image_array = np.array(image)

    return image_array

# slice the sequence
def create_seq_list(seq, size=(15,15)):
    seq_list = []
    
    # append the letter into seq_list (0~1)
    for i in range (len(seq)):
        seq_list.append(create_image_array_from_text(seq[i], size = size))
    
    seq_list = np.array(seq_list)
        
    return seq_list

# apply convolution to img
def convolution(img, conv):
    # if conv is str
    if isinstance(conv, str):
        conv = [conv]
        
    # change conv into lower
    conv = [word.lower() for word in conv]
    
    # if convolution layer is empty
    if conv == ['none']:
        return img
    
    for i in range(len(conv)):
        
        # divide the cases    
        # if convolution layer is Edge Detection 
        if conv[i] == 'edge':
            kernal = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
        
        # if convolution layer is Vertical Edge Detection
        elif conv[i] == 'vedge':
            kernal = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
            
        # if convolution layer is horizontal Edge Detection
        elif conv[i] == 'hedge':
            kernal = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
        
        # if convolution layer is blur    
        elif conv[i] == 'blur':
            kernal = np.array([[1/9,1/9,1/9],
                            [1/9,1/9,1/9],
                            [1/9,1/9,1/9]])
        # if there is no name
        else : 
            raise NameError
        
        # apply convolution into original img
        img = convolve2d(img, kernal, mode = 'same', boundary = 'symm')
    
    return img


# calculate best array that fits
def cal_distance(m1, m2, method):
    
    # first, we have to change the method name using lower()
    method = method.lower()
    
    # calculating distance by method
    # when best-expressing matrix is chosen by Euclidean Distance
    if method == 'euclidean' :
        return np.linalg.norm(m1 - m2)
    
    # when best-expressing matrix is chosen by Cosine Similarity
    elif method == 'cos' : 
        dot_product = np.dot(m1.flatten(), m2.flatten())
        norm_m1 = np.linalg.norm(m1)
        norm_m2 = np.linalg.norm(m2)
    
        similarity = dot_product / (norm_m1 * norm_m2)
        return 1 - similarity   # because we are calculating distance;not the similartiy
    
    # when best-expressing matrix is chosen by Pearson Correlation
    elif method == 'corr' :
        covariance_matrix = np.cov(m1.flatten(), m2.flatten())
        cov = covariance_matrix[0, 1]
        
        std_m1 = np.std(m1)
        std_m2 = np.std(m2)
        
        correlation_coefficient = cov / (std_m1 * std_m2)
        return 1 - correlation_coefficient # because we are calculating distance;not the similartiy
    
    # when best-expressing matrix is chosen by Manhattan Distance
    elif method == 'manhattan' :
        distance = np.sum(np.abs(m1 - m2))
        return distance
    
    # when method name is wrong
    else : 
        raise NameError
    
# find the closest matrix among matrix list
def find_closest_matrix(target_matrix, matrix_list, method = 'Euclidean'):
    closest_matrix = None
    min_distance = float('inf')
    index = 0
    
    # if two matrices have different size
    if target_matrix.shape != matrix_list[0].shape:
        raise Exception('Matrices are imcompatible.')

    for i, candidate_matrix in enumerate(matrix_list):
        distance = cal_distance(target_matrix, candidate_matrix, method)
        
        if distance < min_distance:
            min_distance = distance
            closest_matrix = candidate_matrix
            index = i
            
    # print(min_distance, method, index)

    return index

# slice the img into block, and get the best letter expressing the block
def img2txt(img, size, method = 'Euclidean', conv = 'None'):
    
    if len(img.shape) != 2:
        raise Exception('Wrong dimension!')
    
    rows = math.floor(img.shape[0]/size)
    cols = math.floor(img.shape[1]/size)
    
    pixel = np.empty((rows, cols), dtype=str)
    
    # if there is convolution, apply it before dividing
    img = convolution(img, conv = conv)
    
    # divide the block, and find the letter that best represents it
    for y in range(rows):
        for x in range(cols):
            block = img[y * size: (y + 1) * size, x * size: (x + 1) * size]
            i = find_closest_matrix(block, seq_list, method)
            pixel[y, x] = seq[i]
            
    return pixel

# print full array without loss
def printfullarray(array):
    
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
            print(array[i][j], end = '')
        print()




seq = '!#$%^&*()<>?1234567890-= ■□'    # list of letters
size = 5    # size of block
file_path = r"C:\Users\82103\Downloads\example3.jpeg"     # location on the img

# as you can check, the file is Eiffel Tower
seq_list = create_seq_list(seq, size = (size, size))
img = importimage(file_path)
pixel = img2txt(img, size = size, method = 'manhattan', conv = 'blur')

printfullarray(pixel)