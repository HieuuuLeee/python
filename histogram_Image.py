import matplotlib.pyplot as plt
import numpy as np
import cv2

# Cumulative Distribution Function
g = cv2.imread('image/322868_1100-1100x628.jpg')
hist, bin = np.histogram(g.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_list = list(cdf)
plt.plot(list(range(len(cdf_list))), cdf_list)
plt.xlabel("gray_labels")
plt.ylabel("Cumulatice Image")
plt.title("CDF")
plt.show()

# Histogram Image Function
img = cv2.imread('image/322868_1100-1100x628.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()