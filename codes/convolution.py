import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

filepath = r"C:\Users\82103\Downloads\example.jpg"
img = img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
img_array = np.array(img)

# for example, we will do edge detection
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# apply convolution to img
result = convolve2d(img_array, kernel, mode='same', boundary='symm')

# visualizing
plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Image after Convolution')

plt.show()
