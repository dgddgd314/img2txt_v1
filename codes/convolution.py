import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

# apply convolution to img
def convolution(img, conv = 'None'):
    
    # change conv into lower
    conv = conv.lower()
    
    # divide the cases
    # if convolution layer is Edge Detection
    if conv == 'edge':
        kernal = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
    
    # if there is no name
    else : 
        raise NameError
    
    # apply convolution into original img
    conv_img = convolve2d(img, kernal, mode = 'same', boundary = 'symm')
    
    return conv_img

filepath = r"C:\Users\82103\Downloads\example.jpg"
img = img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
img_array = np.array(img)

result = convolution(img, conv = 'edge')

# visualizing
plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Image after Convolution')

plt.show()

print(result)