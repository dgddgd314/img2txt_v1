from PIL import Image, ImageDraw, ImageFont
import numpy as np

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
    
    # when method name is wrong
    else : 
        raise NameError
    

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
            
    print(min_distance, method, index)

    return closest_matrix

# example

seq = '!@#$%^&*()qwertyuiop[]asdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:"ZXCVBNM<>?1234567890-='
size = (6,6)

seq_list = create_seq_list(seq, size = size)

#print(seq_list)

matrix1 = np.random.rand(6,6)
#matrix1 = create_image_array_from_text('3', size = size)

print(f'{matrix1} : matrix1 \n')

closest_matrix = find_closest_matrix(matrix1, seq_list)

print(closest_matrix)