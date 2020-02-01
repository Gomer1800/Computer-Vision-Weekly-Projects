% 1. read and display image
Landscape = imread('landscape.jpg');
imshow(Landscape)

% 2. Convert image to grayscale
Grayscale = rgb2gray(Landscape);
imshow(Grayscale)
imwrite(Grayscale,'landscape_grayscale.jpg');

% 3. Find max/min intensity values in grayscale
% and spatial coordinates
minimum = min(min(Grayscale))
[x,y]=find(Grayscale==minimum);
Xmin = x(1)
Ymin = y(1)

maximum = max(max(Grayscale))
[x,y]=find(Grayscale==maximum);
Xmax = x(1)
Ymax = y(1)

% Record grayscale image size in bytes
s = dir('landscape_grayscale.jpg');         
landscape_filesize = s.bytes

% Code to shrink resolution