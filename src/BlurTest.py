import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

image = cv.imread('car.jpg')

#gaussian blur
#Kernel size should be positive and odd.
gblr = cv.GaussianBlur(image, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Preview', np.hstack((image, gblr)))
cv.waitKey(0)
cv.destroyAllWindows()

#Extras 
#Plotting the image using matplotlib 
plt.imshow(image)
plt.colorbar()


#Canny edge detection system to pecify your raw image, lower pixel threshold, and higher pixel threshold

edges = cv.Canny(image,100,300)

cv.imshow('Extra', edges)
cv.waitKey(0)
cv.destroyAllWindows()


#Numpy min, max , mean values of the image
min, max, mean = image.min(), image.max(), image.mean()

print(f'DEBUG:min={min}, max={max}, mean={mean}')
