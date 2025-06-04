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
HDR Core Image Aesthetics Module

This module provides comprehensive image aesthetics analysis and modeling capabilities
for HDR images. It includes color palette extraction, composition analysis, and
multidimensional aesthetic modeling tools that help analyze and understand the
visual properties of HDR images.

The module implements advanced machine learning techniques including K-means clustering
for color palette extraction and provides frameworks for extensible aesthetic analysis.

Classes:
    ImageAestheticsModel: Abstract base class for aesthetic modeling
    Palette: Color palette extraction and analysis using K-means clustering
    MultidimensionalImageAestheticsModel: Container for multiple aesthetic models
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import copy, colour, skimage.transform, math, os
import sklearn.cluster, skimage.transform
import numpy as np
import functools
from . import processing, utils, image
import preferences.preferences as pref
from timeit import default_timer as timer



# -----------------------------------------------------------------------------
# --- Class ImageAestheticsModel ----------------------------------------------
# -----------------------------------------------------------------------------
class ImageAestheticsModel():
    """
    Abstract base class for image aesthetics modeling.
    
    This class provides the foundation for all aesthetic analysis models in the
    uHDR system. It defines the basic interface that all aesthetic models should
    implement for consistent integration with the processing pipeline.
    
    Static Methods:
        build: Factory method to create aesthetic model instances
    """
    
    def build(processPipe, **kwargs): 
        """
        Factory method to build an aesthetic model instance.
        
        Args:
            processPipe (hdrCore.processing.ProcessPipe): Processing pipeline
            **kwargs: Additional model-specific parameters
            
        Returns:
            ImageAestheticsModel: Base model instance
        """
        return ImageAestheticsModel()

# -----------------------------------------------------------------------------
# --- Class Palette -----------------------------------------------------------
# -----------------------------------------------------------------------------
class Palette(ImageAestheticsModel):
    """
    Color palette extraction and analysis using advanced clustering techniques.
    
    This class provides sophisticated color palette extraction from HDR images using
    K-means clustering in perceptually uniform color spaces. The extracted palettes
    can be used for aesthetic analysis, color grading guidance, and visual style
    transfer applications.
    
    The palette colors are automatically sorted by their distance from black in the
    specified color space, providing a consistent ordering for analysis and display.
    
    Attributes:
        name (str): Descriptive name for the palette
        colorSpace (colour.models.RGB_COLOURSPACES): Color space of the palette
        nbColors (int): Number of colors in the palette
        colors (numpy.ndarray): Array of color values with shape (nbColors, 3),
                               sorted by distance from black
        type (hdrCore.image.imageType): Type of the source image (SDR/HDR)
        
    Methods:
        createImageOfPalette: Generate a visual representation of the palette
        __repr__: String representation of the palette
        __str__: Human-readable string representation
        
    Static Methods:
        build: Extract color palette from a processing pipeline
    """    
    
    # constructor
    def __init__(self, name, colors, colorSpace, type):
        """
        Initialize a color palette with specified properties.
        
        Args:
            name (str): Name identifier for this palette
            colors (numpy.ndarray): Array of color values with shape (N, 3)
            colorSpace (colour.models.RGB_COLOURSPACES): Color space definition
            type (hdrCore.image.imageType): Source image type
        """
        self.name       = name
        self.colorSpace = colorSpace
        self.nbColors   = colors.shape[0]
        self.colors     = np.asarray(sorted(colors.tolist(), key = lambda u  : np.sqrt(np.dot(u,u))))
        self.type       = type

    @staticmethod    
    def build(processpipe, nbColors=5, method='kmean-Lab', processId=-1, **kwargs):
        """
        Extract a color palette from an image using clustering algorithms.
        
        This method analyzes the image at a specific point in the processing pipeline
        and extracts the most representative colors using advanced clustering techniques.
        The default method uses K-means clustering in the perceptually uniform Lab
        color space for optimal color separation.
        
        Args:
            processpipe (hdrCore.processing.ProcessPipe): Image processing pipeline
            nbColors (int, optional): Number of colors to extract (default: 5)
            method (str, optional): Clustering method to use (default: 'kmean-Lab')
            processId (int, optional): Pipeline stage to analyze (default: -1, final stage)
            **kwargs: Additional method-specific parameters:
                - removeBlack (bool): Whether to exclude dark colors (default: True)
                
        Returns:
            hdrCore.aesthetics.Palette: Extracted color palette with specified number of colors
            
        Note:
            The 'kmean-Lab' method converts the image to Lab color space before clustering
            to ensure perceptually uniform color separation. When removeBlack=True,
            the algorithm extracts nbColors+1 clusters and removes the darkest one.
        """
        # get image according to processId
        image_ = processpipe.processNodes[processId].outputImage

        # according to method
        if method == 'kmean-Lab':
            # taking into acount supplemental parameters of 'kmean-Lab'
            #  'removeblack' : bool
            defaultParams = {'removeBlack': True}
            if 'removeBlack' in kwargs: removeBlack = kwargs['removeBlack']
            else: removeBlack = defaultParams['removeBlack']

            # get image according to processId
            image_ = processpipe.processNodes[processId].outputImage


            # to Lab then to Vector
            imageLab = processing.ColorSpaceTransform().compute(image_,dest='Lab')
            imgLabDataVector = utils.ndarray2vector(imageLab.colorData)

            if removeBlack:
                # k-means: nb cluster = nbColors + 1
                kmeans_cluster_Lab = sklearn.cluster.KMeans(n_clusters=nbColors+1)
                kmeans_cluster_Lab.fit(imgLabDataVector)

                cluster_centers_Lab = kmeans_cluster_Lab.cluster_centers_
                
                # remove darkness one
                idxLmin = np.argmin(cluster_centers_Lab[:,0])                           # idx of darkness
                cluster_centers_Lab = np.delete(cluster_centers_Lab, idxLmin, axis=0)   # remove min from cluster_centers_Lab

            else:
                # k-means: nb cluster = nbColors
                kmeans_cluster_Lab = sklearn.cluster.KMeans(n_clusters=nbColors)
                kmeans_cluster_Lab.fit(imgLabDataVector)
                cluster_centers_Lab = kmeans_cluster_Lab.cluster_centers_

            colors = cluster_centers_Lab
        else: colors = None

        return Palette('Palette_'+image_.name,colors, image.ColorSpace.Lab(), image_.type)

    def createImageOfPalette(self, colorWidth=100):
        """
        Generate a visual representation of the color palette.
        
        Creates an image showing all palette colors as horizontal bands, which is
        useful for visualization and comparison of different palettes. The colors
        are converted to sRGB for display while maintaining proper color management.
        
        Args:
            colorWidth (int, optional): Width in pixels for each color band (default: 100)
            
        Returns:
            hdrCore.image.Image: Visual representation of the palette with colors
                                arranged as horizontal bands
                                
        Note:
            The output image has dimensions (colorWidth, nbColors*colorWidth, 3)
            and is created as an SDR image in sRGB color space for display compatibility.
        """
        if self.colorSpace.name =='Lab':
            if self.type == image.imageType.HDR :
                cRGB = processing.Lab_to_sRGB(self.colors, apply_cctf_encoding=True)
            else:
                cRGB = processing.Lab_to_sRGB(self.colors, apply_cctf_encoding=False)

        elif self.colorSpace.name=='sRGB':
            cRGB = self.colors
        width = colorWidth*cRGB.shape[0]
        height=colorWidth
        # return image
        img = np.ones((height,width,3))

        for i in range(cRGB.shape[0]):
            xMin= i*colorWidth
            xMax= xMin+colorWidth
            yMin=0
            yMax= colorWidth
            img[yMin:yMax, xMin:xMax,0]=cRGB[i,0]
            img[yMin:yMax, xMin:xMax,1]=cRGB[i,1]
            img[yMin:yMax, xMin:xMax,2]=cRGB[i,2]
        # colorData, name, type, linear, colorspace, scalingFactor
        return image.Image(
            '.',self.name,
            img,  
            image.imageType.SDR, False, image.ColorSpace.sRGB())

    # __repr__ and __str__
    def __repr__(self):
        """
        Return a detailed string representation of the palette.
        
        Returns:
            str: Formatted string containing all palette properties including
                 name, color space, number of colors, color values, and image type
        """
        res =   " Palette{ name:"           + self.name                 + "\n"  + \
                "          colorSpace: "    + self.colorSpace.name      + "\n"  + \
                "          nbColors: "      + str(self.nbColors)        + "\n"  + \
                "          colors: \n"      + str(self.colors)          + "\n " + \
                "          type: "          + str(self.type)            + "\n }"  
        return res

    def __str__(self):
        """
        Return a human-readable string representation of the palette.
        
        Returns:
            str: Same as __repr__ for consistency
        """
        return self.__repr__()

# -----------------------------------------------------------------------------
# --- Class MultidimensionalImageAestheticsModel ------------------------------
# -----------------------------------------------------------------------------
class MultidimensionalImageAestheticsModel():
    """
    Container for multiple image aesthetic analysis models.
    
    This class manages a collection of different aesthetic analysis models for
    comprehensive image evaluation. It supports various types of aesthetic analysis
    including color palette extraction, composition analysis through convex hulls,
    and composition strength line detection.
    
    The class provides a unified interface for managing multiple aesthetic models
    and tracking changes in the processing pipeline that might affect the analysis.
    
    Attributes:
        processpipe (hdrCore.processing.ProcessPipe): Associated processing pipeline
        processPipeChanged (bool): Flag indicating if pipeline has been modified
        imageAestheticsModels (dict): Collection of aesthetic models indexed by key
        
    Methods:
        add: Add a new aesthetic model to the collection
        get: Retrieve a specific aesthetic model by key
        build: Create and add a new model using a builder function
    """
    
    def __init__(self, processpipe):
        """
        Initialize the multidimensional aesthetic model container.
        
        Args:
            processpipe (hdrCore.processing.ProcessPipe): Processing pipeline to analyze
        """
        self.processpipe = processpipe
        self.processPipeChanged = True
        self.imageAestheticsModels = {}

    def add(self, key, imageAestheticsModel):
        """
        Add an aesthetic model to the collection.
        
        Args:
            key (str): Unique identifier for the model
            imageAestheticsModel (ImageAestheticsModel): Model instance to add
        """
        self.imageAestheticsModels[key] = imageAestheticsModel

    def get(self, key):
        """
        Retrieve an aesthetic model from the collection.
        
        Args:
            key (str): Identifier of the model to retrieve
            
        Returns:
            ImageAestheticsModel or None: The requested model if found, None otherwise
        """
        iam = None
        if key in self.imageAestheticsModels: iam = self.imageAestheticsModels[key]
        return iam

    def build(self, key, builder, processpipe):
        """
        Create and add new aesthetic models using builder functions.
        
        Args:
            key (str or list): Model identifier(s)
            builder (function or list): Builder function(s) to create models
            processpipe (hdrCore.processing.ProcessPipe): Processing pipeline
            
        Note:
            If key and builder are lists, they should have the same length and
            corresponding elements will be paired for model creation.
        """
        if not isinstance(key,list):
            key, builder = [key],[builder]
        for k in key:
            self.add(k,builder.build(processpipe))
# -----------------------------------------------------------------------------
