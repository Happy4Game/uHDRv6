# uHDR: HDR image editing software
#   Copyright (C) 2021  remi cozot 
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
# hdrCore project 2020
# author: remi.cozot@univ-littoral.fr

# -----------------------------------------------------------------------------
# --- Package hdrCore ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
HDR Core Utilities Module

This module provides essential utility functions for HDR image processing, including
file operations, array manipulations, and mathematical transformations used throughout
the hdrCore package.

Functions:
    - filenamesplit: Parse filename into path, name, and extension components
    - filterlistdir: Filter directory contents by file extensions
    - ndarray2vector: Convert 2D image arrays to 1D vectors
    - NPlinearWeightMask: Generate linear weight masks for image blending
    - croppRotated: Calculate crop dimensions for rotated images
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import os, math
import numpy as np

# -----------------------------------------------------------------------------
# --- Package functions -------------------------------------------------------
# -----------------------------------------------------------------------------
def filenamesplit(filename):
    """
    Parse a filename into its constituent path, name, and extension components.

    Args:
        filename (str): Complete filename including path and extension
            
    Returns:
        tuple: A tuple containing (path, name, extension) where:
            - path (str): Directory path to the file
            - name (str): Filename without extension
            - extension (str): File extension in lowercase
            
    Example:
        >>> filenamesplit("./dir0/dir1/name.ext")
        ("./dir0/dir1/", "name", "ext")
    """
    
    path, nameWithExt = os.path.split(filename)
    splits = nameWithExt.split('.')
    ext = splits[-1].lower()
    name = '.'.join(splits[:-1])
    return (path,name,ext)

def filterlistdir(path, extList):
    """
    Filter directory contents to return only files with specified extensions.

    Args:
        path (str): Path to the directory to scan
        extList (str, list, or tuple): File extension(s) to filter by
            
    Returns:
        list: Sorted list of filenames matching the specified extensions
           
    Example:
        >>> filterlistdir('./images/', ('.jpg', '.JPG', '.png'))
        ['image1.jpg', 'image2.png', 'photo.JPG']
    """ 
    
    ext = None
    if isinstance(extList, list):
        ext = tuple(extList)
    elif isinstance(extList, str):
        ext = extList
    elif isinstance(extList, tuple):
        ext = extList
    filenames = sorted(os.listdir(path))
    res = list(filter(lambda x: x.endswith(ext),filenames))

    return res

def ndarray2vector(nda):
    """
    Transform a 2D or 3D image array into a 1D vector format.

    This function reshapes image data from spatial dimensions (height, width, channels)
    to a linear vector format (pixels, channels) for processing operations.

    Args:
        nda (numpy.ndarray): Input image array, either 2D (grayscale) or 3D (color)
            
    Returns:
        numpy.ndarray: Reshaped 1D array where each row represents a pixel
    """
    if len(nda.shape) ==2 :
        x,y = nda.shape
        c = 1
    elif len(nda.shape) ==3:
        x,y,c = nda.shape
    return np.reshape(nda, (x * y, c))


# ------------------------------------------------------------------------------------------
#def linearWeightMask(x, xMin, xMax, xTolerance):
#    if x < (xMin - xTolerance):     y = 0
#    elif x <= xMin:                 y = (x -(xMin - xTolerance))/xTolerance
#    elif x <= xMax :                y = 1
#    elif x <= (xMax + xTolerance):  y = 1 - (x - xMax)/xTolerance
#    else:                           y = 0
#    return y

# ------------------------------------------------------------------------------------------

def NPlinearWeightMask(x, xMin, xMax, xTolerance):
    """
    Generate a linear weight mask with smooth transitions at boundaries.

    Creates a trapezoidal weight function that transitions smoothly from 0 to 1
    within the specified tolerance regions. Used for smooth image blending and
    masking operations.

    Args:
        x (numpy.ndarray): Input 2D array of values to generate mask for
        xMin (float): Lower bound where mask transitions from 0 to 1
        xMax (float): Upper bound where mask transitions from 1 to 0
        xTolerance (float): Width of transition regions
            
    Returns:
        numpy.ndarray: 2D weight mask with same dimensions as input x
        
    Note:
        The mask has the following behavior:
        - Values below (xMin - xTolerance): weight = 0
        - Values between (xMin - xTolerance) and xMin: linear transition 0→1
        - Values between xMin and xMax: weight = 1
        - Values between xMax and (xMax + xTolerance): linear transition 1→0
        - Values above (xMax + xTolerance): weight = 0
    """
    # reshape x
    h,w  = x.shape  # 2D array
    xv = np.reshape(x,(h*w,1))
    y = np.ones((h*w,1))
    y = np.where((xv <= (xMin - xTolerance)),               0,y)                                        # (0)                +-----------+
    y = np.where((xv > (xMin - xTolerance))&(xv <= xMin),   (xv -(xMin - xTolerance))/xTolerance,y)     # (1)               /             \
    y = np.where((xv > (xMin))&(xv <= xMax),                1,y)                                        # (2)              /               \
    y = np.where((xv > (xMax))&(xv <= xMax + xTolerance),   1 - (xv - xMax)/xTolerance,y)               # (3)     --------+                 +-------
    y = np.where((xv > (xMax + xTolerance)),                0,y)                                        # (4)         (0)   (1)  (2)    (3)    (4)

    return np.reshape(y,(h,w))

def croppRotated(h, w, alpha):
    """
    Calculate the crop dimensions for a rotated image to avoid black borders.

    When an image is rotated by an arbitrary angle, black borders appear at the edges.
    This function calculates the maximum rectangular crop that fits within the rotated
    image bounds without including any black borders.

    Args:
        h (float): Original image height
        w (float): Original image width
        alpha (float): Rotation angle in degrees
            
    Returns:
        tuple: (new_height, new_width) of the maximum crop rectangle
        
    Note:
        The calculation assumes rotation around the image center and uses
        trigonometric relationships to find the largest inscribed rectangle.
    """

    cosA = math.cos(math.radians(alpha))
    sinA = math.sin(math.radians(alpha))

    v_up_left = - h/(h*cosA + w*sinA)
    v_up_right = h/(h*cosA - w*sinA)

    v = min(abs(v_up_left),abs(v_up_right))

    return (h*v, w*v)

# ------------------------------------------------------------------------------------------
# ---- Constants ---------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#######################
#HDRdisplay = {
#    'none' :                {'scaling':1,  'post':'',                       'tag':None},
#    'vesaDisplayHDR1000' :  {'scaling':12, 'post':'_vesa_DISPLAY_HDR_1000', 'tag':'vesaDisplayHDR1000'}
#    }
# ------------------------------------------------------------------------------------------
