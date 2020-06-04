import cv2
import numpy as np
import matplotlib.pyplot as plt

def Noise_EdgeFilter(string):
    img = cv2.imread(string)
    
    img_origin = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    plt.subplot(221),
    plt.imshow(img_origin, cmap='gray'),
    plt.title('Origin')
    plt.xticks([]), plt.yticks([])

    ddepth = cv2.CV_16S
    kernel_size = 3
    
    lapla_img = cv2.Laplacian(img_origin, ddepth, ksize = kernel_size)
    
    plt.subplot(222),
    plt.imshow(cv2.convertScaleAbs(lapla_img), cmap='gray'),
    plt.title('Lablacian')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(223),
    plt.imshow(cv2.convertScaleAbs(lapla_img) + img_origin, cmap='gray'),
    plt.title('Lablacian filter')
    plt.xticks([]), plt.yticks([])
    
    img_gaussian = cv2.GaussianBlur(img, (3,3), 0)
    
    plt.subplot(224),
    plt.imshow(img_gaussian, cmap='gray'),
    plt.title('Gaussian')
    plt.xticks([]), plt.yticks([])
    
Noise_EdgeFilter('image/322868_1100-1100x628.jpg')