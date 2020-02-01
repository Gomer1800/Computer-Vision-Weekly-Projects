#!/usr/bin/env python
# coding: utf-8

# In[72]:


import sys
import os
import math
import numpy as np

from scipy import ndimage as ndi
from skimage.transform import hough_line, hough_line_peaks
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib import pyplot as plt


# In[73]:


# Function: Get image dimensions
def get_size( my_image ):
    img = Image.open(my_image)
    x,y = img.size
    return (x,y)


# In[74]:


# Function: Generate Gray Scale Numpy Array from Image
def create_grayscale( my_image ):
    grayscale = np.array(Image.open( my_image ).convert('L'))
    return grayscale


# In[75]:


# Function: Create grayscale and show image
def show_grayscale( my_image ):
    plt.figure(figsize = (10,10))
    plt.imshow(my_image, cmap = plt.get_cmap(name = 'gray'))
    plt.show()


# In[76]:


# Function: LoG
def LoG (my_image, sigma):
    A = gaussian_filter(grayscale, sigma=math.sqrt(sigma)/math.sqrt(2))
    B = gaussian_filter(grayscale, sigma=math.sqrt(sigma)*math.sqrt(2))
    return np.subtract(A,B)


# In[77]:


#Gradient = [Gx,Gy]
def Gradient( pixel_arr ):
    #padding
    padded_arr = np.pad(pixel_arr,1)

    #Sx
    Sx_prewitt = [[-1,-1,-1],[0,0,0],[1,1,1]]

    #Sy
    Sy_prewitt = [[-1,0,1],[-1,0,1],[-1,0,1]]

    Gx = []
    Gy = []
    for i in range(len(pixel_arr)):
        row_x = []
        row_y = []
        for j in range(len(pixel_arr[0])):
            #get neighboring pixels 
            neigh_matrix = padded_arr[i:i+3,j:j+3]

            # apply filter Sx: Sx_prewitt or Sx_sobel
            # Sy: Sy_prewitt or Sy_sobel 
            g_x = np.multiply(neigh_matrix,Sx_prewitt)
            g_y = np.multiply(neigh_matrix,Sy_prewitt)
                  
            #sum the matrix values and place it back to same
            # position of the original (middle) pixel 
            sum_x = np.sum(g_x)
            sum_y = np.sum(g_y)
            row_x.append(sum_x)
            row_y.append(sum_y)
            
    Gx.append(row_x)
    Gy.append(row_y)

    #square Gx and Gy 
    Gx2 = np.multiply(Gx,Gx)
    Gy2 = np.multiply(Gy,Gy)
    
    #sum of the two squares 
    sum_Gx2 = Gx2 + Gy2

    #square root to get the gradient magnitude 
    gradient_mag = np.sqrt(sum_Gx2)
    print(gradient_mag)

    #apply threshold
    gradient_mag[gradient_mag < 60] = 0

    #show the applied filter image
    edge_img = Image.fromarray(gradient_mag)
    show_grayscale(edge_img)
    return edge_img


# In[84]:


def generate_Hough_Transform(my_image):
    """
    From this url:
    https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
    
    The Hough transform constructs a histogram array representing the parameter
    space (i.e., an M×N matrix, for M different values of the radius
    and N different values of θ). For each parameter combination, r and θ,
    we then find the number of non-zero pixels in the input image that
    would fall close to the corresponding line, and increment the array at
    position (r,θ) appropriately.

    We can think of each non-zero pixel “voting” for potential line candidates.
    The local maxima in the resulting histogram indicates the parameters of
    the most probably lines. In our example, the maxima occur at 45 and 135
    degrees, corresponding to the normal vector angles of each line.
    """
    
    # Generate Hough lines
    h, theta, d = hough_line(my_image)

    # Generate Figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    # Figure 1: original image
    ax[0].imshow(my_image, cmap = plt.get_cmap(name = 'gray'))
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    # Figure 2: Hough Lines
    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap = plt.get_cmap(name = 'gray'), aspect=1/1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('on')

    # Figure 3: Superimpose Hough Lines
    ax[2].imshow(my_image, cmap = plt.get_cmap(name = 'gray'))
    origin = np.array((0, grayscale.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((my_image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.show()


# In[85]:


grayscale = create_grayscale('bike-lane.jpg')
show_grayscale(grayscale)
(size_x, size_y) =  get_size('bike-lane.jpg')

LoG_Image = LoG( grayscale, sigma=5)
generate_Hough_Transform(LoG_Image)


# In[81]:


grayscale = create_grayscale('building.gif')
show_grayscale(grayscale)
(size_x, size_y) =  get_size('building.gif')

LoG_Image = LoG( grayscale, sigma=5)
generate_Hough_Transform(LoG_Image)


# In[82]:


grayscale = create_grayscale('corner_window.jpg')
show_grayscale(grayscale)
(size_x, size_y) =  get_size('corner_window.jpg')

LoG_Image = LoG( grayscale, sigma=5)
generate_Hough_Transform(LoG_Image)


# In[83]:


grayscale = create_grayscale('corridor.jpg')
show_grayscale(grayscale)
(size_x, size_y) =  get_size('corridor.jpg')

LoG_Image = LoG( grayscale, sigma=5)
generate_Hough_Transform(LoG_Image)


# In[26]:


grayscale = create_grayscale('New York City.jpg')
show_grayscale(grayscale)
(size_x, size_y) =  get_size('New York City.jpg')

LoG_Image = LoG( grayscale, sigma=5)
generate_Hough_Transform(LoG_Image)

