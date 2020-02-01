#!/usr/bin/env python
# coding: utf-8

# In[50]:


import sys
import os
import math
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib import pyplot as plt


# In[4]:


# Get image dimensions
def get_size( my_image ):
    img = Image.open(my_image)
    x,y = img.size
    return (x,y)


# In[5]:


# Generate Gray Scale Numpy Array from Image
def create_grayscale( my_image ):
    grayscale = np.array(Image.open( my_image ).convert('L'))
    return grayscale


# In[6]:


# Create grayscale and show image
def show_grayscale( my_image ):
    plt.figure(figsize = (10,10))
    plt.imshow(my_image, cmap = plt.get_cmap(name = 'gray'))
    plt.show()


# In[40]:


# LoG Function
def LoG (my_image, sigma):
    A = gaussian_filter(grayscale, sigma=math.sqrt(sigma)/math.sqrt(2))
    B = gaussian_filter(grayscale, sigma=math.sqrt(sigma)*math.sqrt(2))
    return np.subtract(A,B)


# In[20]:


grayscale = create_grayscale('bike-lane.jpg')
show_grayscale(grayscale)

print("{} {}".format("Array Size = ", grayscale.size))

(size_x, size_y) =  get_size('bike-lane.jpg')

print("{} {}".format("Size x = ", size_x))
print("{} {}".format("Size y = ", size_y))


# In[ ]:


##iterate through grayscale and apply LoG
#for y in range(size_y):
#    for x in range(size_x):
#        my_array[y,x] = LoG(x,y,sigma_squared)

#print("{} {}".format("Array Size = ", my_array.size))


# In[43]:


# Sigma^2 = 1

C = LoG( grayscale, sigma=1)
show_grayscale(C)


# In[44]:


# Sigma^2 = 2

C = LoG( grayscale, sigma=2)
show_grayscale(C)


# In[45]:


# Sigma^2 = 5

C = LoG( grayscale, sigma=5)
show_grayscale(C)


# In[ ]:




