#!/usr/bin/env python
# coding: utf-8

# In[163]:


from PIL import Image, ImageStat          # Python Imaging Library
from skimage import data
from skimage.feature import match_template

import PIL.ImageOps 
import numpy as np              # Numerical Python
import matplotlib.pyplot as plt # Python plotting


# In[132]:


# Helper Functions

# show grayscale image
def show_grayscale( my_image ):
    plt.figure(figsize = (10,10))
    plt.imshow(my_image, cmap = plt.get_cmap(name = 'gray'))
    plt.show()
    

# Generate Gray Scale Numpy Array from Image
def create_grayscale( my_image ):
    grayscale = np.array(Image.open( my_image ).convert('L'))
    show_grayscale(grayscale)
    return grayscale

# apply normalized correlation, rock
def norm_correlate( reference, template ):
    result = match_template(reference, template,pad_input=True)
    result = np.round(result, 3)
    result = result + 1
    result = result * 127
    res_img = Image.fromarray(result)

    plt.figure( figsize = (10,10))
    plt.imshow(res_img, cmap = plt.get_cmap(name = 'gray'))
    plt.show()
    return result
    
#apply threshold 
def threshold_this( image, maximum ):
    image[image < maximum] = 0

    res_img = Image.fromarray(image)
    plt.figure( figsize = (10,10))
    plt.imshow(res_img, cmap = plt.get_cmap(name = 'gray'))
    plt.show()
    
#threshold rock template at 102
def create_binary( image, threshold, invert):
    bi = np.asarray(image,dtype=np.uint8)
    
    if invert==False:
        bi[bi < threshold] = 0
        bi[bi > threshold] = 255
    else:
        bi[bi < threshold] = 255
        bi[bi > threshold] = 0
        
    bi_img = Image.fromarray(bi, 'L')
    
    # show binary image
    plt.figure( figsize = (10,10))
    plt.imshow(bi_img, cmap = plt.get_cmap(name = 'gray'))
    plt.show()
    return bi


# In[144]:


#Experiment 1, Simple Detection

# Convert all images to grayscale
reference_hands = create_grayscale( 'Hands.jpg')
template_rock = create_grayscale( 'Token_Rock_2.jpg')
template_paper = create_grayscale( 'Token_Paper.jpg')
template_scissors = create_grayscale( 'Token_Scissors.jpg')

#convert images to numpy array
hands    = np.asarray(reference_hands,dtype=np.float32)
rock     = np.asarray(template_rock,dtype=np.float32)
paper    = np.asarray(template_paper,dtype=np.float32)
scissors = np.asarray(template_scissors,dtype=np.float32) 

# get normalized correlation
matched_rock     = norm_correlate(hands, rock)
matched_paper    = norm_correlate(hands, paper)
matched_scissors = norm_correlate(hands, scissors)

# threshold
threshold_this(matched_rock, 205)
threshold_this(matched_paper, 200)
threshold_this(matched_scissors, 200)


# In[168]:


#Experiment 2.1, Simple Detection using Binary Templates

# Convert all images to grayscale
reference_hands = create_grayscale( 'Hands.jpg')
template_rock = create_grayscale( 'Token_Rock_2.jpg')
template_paper = create_grayscale( 'Token_Paper.jpg')
template_scissors = create_grayscale( 'Token_Scissors.jpg')

# Get Reference Image
hands    = np.asarray(reference_hands,dtype=np.float32)

# Get Binary templates 155
rock     = create_binary(template_rock, 190, invert=False)
paper    = create_binary(template_paper, 190, invert=False)
scissors = create_binary(template_scissors, 190, invert=False) 

# get normalized correlation
matched_rock     = norm_correlate(hands, rock)
matched_paper    = norm_correlate(hands, paper)
matched_scissors = norm_correlate(hands, scissors)

# threshold
threshold_this(matched_rock, 205)
threshold_this(matched_paper, 200) #155)
threshold_this(matched_scissors, 200) #190)


# In[193]:


#Experiment 2.1, Simple Detection using Binary Templates

# Convert all images to grayscale
reference_hands = create_grayscale( 'Hands_1.jpg')
template_rock = create_grayscale( 'Token_Rock_2.jpg')
template_paper = create_grayscale( 'Token_Paper.jpg')
template_scissors = create_grayscale( 'Token_Scissors.jpg')

# Get Reference Image
hands    = np.asarray(reference_hands,dtype=np.float32)

# Get Binary templates 155
rock     = create_binary(template_rock, 190, invert=False)
paper    = create_binary(template_paper, 190, invert=False)
scissors = create_binary(template_scissors, 190, invert=False) 

# get normalized correlation
matched_rock     = norm_correlate(hands, rock)
matched_paper    = norm_correlate(hands, paper)
matched_scissors = norm_correlate(hands, scissors)

# threshold
threshold_this(matched_rock, 165)
threshold_this(matched_paper, 165) #155)
threshold_this(matched_scissors, 165) #190)


# In[189]:


#Experiment 4 Simple Detection using Binary Templates on Dark Hand Image

# Convert all images to grayscale
reference_hands = create_grayscale( 'Hands_2.jpg')
template_rock = create_grayscale( 'Token_Rock_2.jpg')
template_paper = create_grayscale( 'Token_Paper.jpg')
template_scissors = create_grayscale( 'Token_Scissors.jpg')

# Get Reference Image
hands    = np.asarray(reference_hands,dtype=np.float32)

# Get Binary templates 155
rock     = create_binary(template_rock, 190, invert=False)
paper    = create_binary(template_paper, 190, invert=False)
scissors = create_binary(template_scissors, 190, invert=False)

# get normalized correlation
matched_rock     = norm_correlate(hands, rock)
matched_paper    = norm_correlate(hands, paper)
matched_scissors = norm_correlate(hands, scissors)

# threshold
threshold_this(matched_rock, 165)
threshold_this(matched_paper, 165)
threshold_this(matched_scissors, 165)


# In[ ]:




