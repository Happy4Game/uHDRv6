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
HDR Core Image Module

This module provides the fundamental image classes and utilities for HDR image processing
within the uHDR system. It includes comprehensive image data structures, color space
management, histogram computation, and image manipulation capabilities.

The module supports multiple image types (SDR, RAW, HDR) and provides robust color space
conversion and channel management. It integrates with various image formats and metadata
systems for complete HDR imaging workflows.

Classes:
    - imageType: Enumeration for different image types (SDR, ARW, HDR)
    - channel: Channel identification and color space management
    - Image: Core image data structure with processing capabilities
    - ColorSpace: Color space definitions and transformations
    - Histogram: Image histogram computation and analysis

Key Features:
    - Multi-format image loading (HDR, RAW, JPEG)
    - Color space conversions (sRGB, XYZ, Lab, LCH)
    - HDR-specific processing and metadata handling
    - Thumbnail generation and caching
    - Dynamic range computation
    - Image splitting and merging operations
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import enum, rawpy, colour, imageio, copy, os, functools, skimage.transform
import numpy as np
from . import utils, processing, metadata
import preferences.preferences as pref

imageio.plugins.freeimage.download()

# -----------------------------------------------------------------------------
# --- Class imageType --------------------------------------------------------
# -----------------------------------------------------------------------------
class imageType(enum.Enum):
    """
    Enumeration for different image file types supported by uHDR.
    
    This enumeration defines the standard image types that can be processed
    within the HDR imaging pipeline, each with specific handling requirements.
    
    Attributes:
        - SDR (int): Standard Dynamic Range images (.jpg, .png)
        - ARW (int): Sony RAW image files (.arw)
        - HDR (int): High Dynamic Range images (.hdr, .exr)
    """

    SDR = 0 # SDR image:                (.jpg)
    ARW = 1 # raw image file: sony ARW  (.arw)
    HDR = 2 # hdr file:                 (.hdr)

# -----------------------------------------------------------------------------
# --- Class channel ----------------------------------------------------------
# -----------------------------------------------------------------------------
class channel(enum.Enum):
    """
    Channel identification and color space management enumeration.
    
    This class provides constants and methods for managing individual color channels
    across different color spaces (sRGB, XYZ, Lab). It supports both single channel
    access and full color space operations.
    
    Attributes:
        - sR, sG, sB (int): Individual sRGB color channels (0, 1, 2)
        - sRGB (int): Full sRGB color space identifier (3)
        - X, Y, Z (int): Individual XYZ color channels (4, 5, 6)
        - XYZ (int): Full XYZ color space identifier (7)
        - L, a, b (int): Individual Lab color channels (8, 9, 10)
        - Lab (int): Full Lab color space identifier (11)
    """
    
    sR      = 0
    sG      = 1
    sB      = 2

    sRGB    = 3

    X       = 4
    Y       = 5
    Z       = 6

    XYZ     = 7

    L       = 8
    a       = 9
    b       = 10

    Lab     = 11

    def colorSpace(self):
        """
        Retrieve the color space name for this channel.
        
        Returns:
            str: Color space type ('sRGB', 'XYZ', or 'Lab')
        """
        csIdx = self.value // 4
        res = None
        if csIdx   == 0:    res ='sRGB'
        elif csIdx == 1:    res = 'XYZ'
        elif csIdx == 2:    res = 'Lab'

        return res

    def getValue(self):
        """
        Retrieve the channel index within its color space.
        
        Returns:
            int: Channel index (0-3) within the color space
        """
        return self.value % 4

    @staticmethod
    def toChannel(s):
        """
        Convert a channel name string to its corresponding channel constant.
        
        Args:
            s (str): Channel name ('sR', 'sG', 'sB', 'X', 'Y', 'Z', 'L', 'a', 'b')
            
        Returns:
            channel: Corresponding channel enumeration value
        """
        if s=='sR' :    return channel.sR
        elif s=='sG' :  return channel.sG
        elif s=='sB' :  return channel.sB

        elif s=='X' :   return channel.X
        elif s=='Y' :   return channel.Y
        elif s=='Z' :   return channel.Z

        elif s=='L' :   return channel.L
        elif s=='a' :   return channel.a
        elif s=='b' :   return channel.b

        else:           return channel.L       
# -----------------------------------------------------------------------------        
# --- Class Image ------------------------------------------------------------
# -----------------------------------------------------------------------------
class Image(object):
    """
    Core HDR image data structure with comprehensive processing capabilities.
    
    This class represents the fundamental image object in the uHDR system, containing
    pixel data, metadata, and all necessary information for HDR image processing.
    It supports various image formats, color spaces, and provides methods for
    image manipulation, analysis, and processing pipeline integration.
    
    Attributes:
        - path (str): Directory path to the image file
        - name (str): Image filename with extension
        - colorData (numpy.ndarray): Image pixel data as 3D array (height, width, channels)
        - shape (tuple): Image dimensions (height, width, channels)
        - type (imageType): Image type classification (SDR, ARW, HDR)
        - linear (bool): Whether image data is in linear color space
        - colorSpace (colour.models.RGB_COLOURSPACES): Current color space definition
        - scalingFactor (float): Scaling factor for normalizing to [0,1] range
        - metadata (hdrCore.metadata.metadata): Associated metadata object
        - histogram (Histogram): Image histogram data
    """

    def __init__(self, path, name, colorData, type, linear, colorspace, scalingFactor=1.0):
        """
        Initialize a new Image object with specified parameters.
            
        Args:
            path (str): Directory path to the image file
            name (str): Image filename with extension
            colorData (numpy.ndarray): Pixel data array with shape (height, width, channels)
            type (imageType): Image type classification
            linear (bool): True if image data is in linear color space
            colorspace (colour.models.RGB_COLOURSPACES): Color space definition
            scalingFactor (float, optional): Scaling factor for [0,1] normalization (default: 1.0)
        """

        self.path           = path                          # path to file          (str)
        self.name           = name                          # filename              (str)
        self.colorData      = colorData                     # float color data      (numpy.ndarray)
        self.shape          = self.colorData.shape          # image size height,width, color channel number (tuple)
        self.type           = type                          # image type            (hdrCore.image.imageType)
        self.linear         = linear                        # image in RGB linear   (bool)
        self.colorSpace     = colorspace                    # colorSpace            (colour.models.RGB_COLOURSPACES)
        self.scalingFactor  = scalingFactor                 # scaling to factor to range [0,1] (float)
        self.metadata       = None                          # associated meta data  (hdrCore.metadata.metadata)   
        self.histogram      = None                          # histogram             (hdrCore.image.Histogram)

    def isHDR(self):
        """
        Check if the image is of HDR type.
        
        Returns:
            bool: True if the image is HDR, False otherwise
        """
        return self.type == imageType.HDR

    def process(self, process, **kwargs):
        """
        Apply a processing operation to the image.
        
        This method applies a Processing object to the current image and returns
        a new processed image. The processing operation is defined by the process
        parameter and its compute method.
        
        Args:
            process (Processing): Processing object with compute method
            **kwargs: Additional parameters passed to the processing operation
                
        Returns:
            Image: New image object with processing applied
        """
        return process.compute(self,**kwargs)

    @staticmethod
    def read(filename, thumb = False):
        """
        Load an image from file with automatic format detection.
        
        This method provides comprehensive image loading with support for various
        formats including HDR (.hdr), RAW (.arw), and standard formats (.jpg).
        It handles thumbnail generation, metadata extraction, and color space
        configuration automatically.
        
        Args:
            filename (str): Complete path to the image file
            thumb (bool, optional): Whether to load/create thumbnail version (default: False)
                
        Returns:
            Image: Loaded image object with metadata and proper color space configuration
            
        Note:
            - For HDR images: Creates thumbnails in ./thumbnails/ directory when requested
            - For RAW images: Uses rawpy with sRGB output and camera white balance
            - Automatically loads existing metadata from .json files if available
        """

        imgDouble, imgDoubleFull = None, None
        # image name
        path, name, ext = utils.filenamesplit(filename)

        # load raw file using rawpy
        if ext=="arw":
            outBit = 16
            raw = rawpy.imread(filename)
            ppParams = rawpy.Params(demosaic_algorithm=None, half_size=False, 
                                            four_color_rgb=False, dcb_iterations=0, 
                                            dcb_enhance=False, fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off, 
                                            noise_thr=None, median_filter_passes=0, 
                                            use_camera_wb=True,                                 # default False
                                            use_auto_wb=False, 
                                            user_wb=None, 
                                            output_color=rawpy.ColorSpace.sRGB,                 # output in SRGB
                                            output_bps=outBit,                                  # default 8
                                            user_flip=None, user_black=None, 
                                            user_sat=None, no_auto_bright=False, 
                                            auto_bright_thr=None, adjust_maximum_thr=0.75, 
                                            bright=1.0, highlight_mode=rawpy.HighlightMode.Clip, 
                                            exp_shift=None, exp_preserve_highlights=0.0, 
                                            no_auto_scale=False,
                                            gamma=None,                                         # linear output
                                            chromatic_aberration=None, bad_pixels_path=None)
            imgDouble = colour.utilities.as_float_array(raw.postprocess(ppParams))/(pow(2,16)-1)
            raw.close()
            scalingFactor, type, linear = 1.0, imageType.ARW, True

        # load jpg, tiff, hdr file using colour
        elif ext=="jpg":
            imgDouble = colour.read_image(filename, bit_depth='float32', method='Imageio')
            scalingFactor, type, linear = 1.0, imageType.SDR, False

        # post processing for HDR scaling to [ ,1]
        elif ext =="hdr":
            if thumb: 
                # do not read input only the thumbnail
                searchStr = os.path.join(path,"thumbnails","_"+name+"."+ext)
                if os.path.exists(searchStr): 
                    imgDouble = colour.read_image(searchStr, bit_depth='float32', method='Imageio') # <--- read thumbnail of input file

                else:
                    if not os.path.exists(os.path.join(path,"thumbnails")): os.mkdir(os.path.join(path,"thumbnails"))

                    # read image and create thumbnail
                    imgDouble = colour.read_image(filename, bit_depth='float32', method='Imageio') # <--- read input file

                    # resize to thumbnail size
                    iY, iX, _ = imgDouble.shape
                    maxX = processing.ProcessPipe.maxSize
                    factor = maxX/iX
                    imgDoubleFull = copy.deepcopy(imgDouble)
                    imgThumbnail =  skimage.transform.resize(imgDouble, (int(iY * factor),maxX ))
                    # save thumbnail
                    colour.write_image(imgThumbnail,searchStr, method='Imageio')

                    imgDouble = imgThumbnail

            else:
                # thumb set to False, read input not the thumbnail
                imgDouble = colour.read_image(filename, bit_depth='float32', method='Imageio')

            type = imageType.HDR
            linear = True
            scalingFactor = 1.0

        # create image object
        res =  Image(path, name+'.'+ext, np.float32(imgDouble),type, linear, None, scalingFactor)           # colorspace = None will be set in metadata.metadata.build(res)
        res.metadata = metadata.metadata.build(res)                                                         # build metadata (read if json file exists, else recover from exif data)

        # update path
        res.metadata.metadata['path'] = copy.deepcopy(path)
        # update size
        if thumb and isinstance(imgDoubleFull, np.ndarray):
            h,w, c = imgDoubleFull.shape
            res.metadata.metadata['exif']['Image Width']    = w
            res.metadata.metadata['exif']['Image Height']   = h

        # update image.colorSpace from metadata
        RGBcolorspace = ColorSpace.sRGB() # delfault color space
        if not ('sRGB' in res.metadata.metadata['exif']['Color Space']):
            res.colorSpace = RGBcolorspace 
        else: # 'sRGB' in color space <from exif>
            res.colorSpace = RGBcolorspace 

        # display ready image
        if 'display' in res.metadata.metadata.keys():
            disp = res.metadata.metadata['display']
            if disp in pref.getHDRdisplays().keys():
                scaling = pref.getHDRdisplays()[disp]['scaling']
                res.colorData = res.colorData/scaling  

        return res

    def write(self,filename):
        """
        Save the image and its metadata to disk.
        
        This method saves HDR images to disk along with their associated metadata
        in a JSON file. It updates the image's path and filename information
        before saving.

        Args:
            filename (str): Target filename for saving the image
            
        Note:
            Only HDR images can be written using this method. The metadata
            is automatically saved to a corresponding .json file.
        """
        if self.isHDR():

            path, name, ext = utils.filenamesplit(filename)
            colour.write_image(self.colorData,filename, method='Imageio')

            # update filename related metadata before saving
            self.name = name+'.'+ext
            self.path  = path
            self.metadata.image = self
            self.metadata.metadata['filename'] = name+'.'+ext
            self.metadata.metadata['path'] =     path

            self.metadata.save()

    @staticmethod
    def toOne(colorData):
        """
        Scale image color data to [0, 1] range based on maximum RGB values.
        
        This utility method normalizes image data by finding the maximum value
        across all RGB channels and scaling accordingly.

        Args:
            colorData (numpy.ndarray): Input image color data
                
        Returns:
            tuple: (scaled_color_data, scaling_factor)
                - scaled_color_data: Normalized image data in [0,1] range
                - scaling_factor: Applied scaling factor (1/max_value)
        """

        imgVector = utils.ndarray2vector(colorData)
        R, G, B = imgVector[:,0], imgVector[:,1], imgVector[:,2]
        maxRGB = max([np.amax(R), np.amax(G), np.amax(B)]) 

        return colorData/maxRGB, 1.0/maxRGB

    def getChannel(self,channel):
        """
        Extract a specific color channel from the image.
        
        This method extracts individual color channels (R, G, B, X, Y, Z, L, a, b)
        by converting to the appropriate color space and returning the requested channel.

        Args:
            channel (channel): Channel identifier (e.g., channel.sR, channel.Y, channel.L)
        
        Returns:
            numpy.ndarray or None: 2D array containing the channel data, or None for invalid channels
        """

        # take into account colorSpace
        destColor = channel.colorSpace()
        image = processing.ColorSpaceTransform().compute(self,dest=destColor)

        if channel.getValue() <3:
            return image.colorData[:,:,channel.getValue()]
        else:
            return None

    def getDynamicRange(self,percentile=None):
        """
        Calculate the dynamic range of the image in stops.
        
        Computes the dynamic range by analyzing the Y (luminance) channel
        and calculating the log2 ratio between maximum and minimum values.

        Args:
            percentile (float, optional): Percentile for robust min/max calculation.
                                        If None, uses actual min/max excluding zeros.
                
        Returns:
            float: Dynamic range in stops (log2 scale)
        """

        Y_min,Y_max = None, None
        Y = self.getChannel(channel.Y)

        if percentile == None : Y_min, Y_max = np.amin(Y[Y>0]), np.amax(Y)                  # use min and max
        else: Y_min, Y_max = np.percentile(Y[Y>0],percentile), np.percentile(Y,100-percentile)   # percentile

        return np.log2(Y_max)-np.log2(Y_min)

    def buildHistogram(self,channel):
        """
        Generate a histogram for the specified image channel.
        
        Creates and stores a histogram object for the given channel, which can
        be used for analysis and display purposes.

        Args:
            channel (channel): Channel to analyze for histogram generation
        """
        self.histogram = Histogram.build(self, channel, nbBins=100, range= None, logSpace = self.isHDR())

    def plot(self,ax,displayTitle=False,title=None,forceToneMapping=True,TMO=None):
        """
        Display the image on a matplotlib axis with optional tone mapping.
        
        This method handles proper display of both SDR and HDR images, applying
        tone mapping when necessary for HDR content.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axis for displaying the image
            displayTitle (bool, optional): Whether to show title (default: False)
            title (str, optional): Custom title text (default: auto-generated)
            forceToneMapping (bool, optional): Apply tone mapping to HDR images (default: True)
            TMO (Processing, optional): Custom tone mapping operator (default: CCTF encoding)
        """
        # default values
        if not title: title= self.name+"("+self.colorSpace.name +" | "+ str(self.type)+")"
        if not TMO: TMO = processing.tmo_cctf()
        if (not (self.type == imageType.HDR)) or (not  forceToneMapping): 
            ax.imshow(self.colorData)
        else: # HDR and forceToneMapping
            ax.imshow(self.process(TMO).process(processing.clip()).colorData)
            title = title+"[auto TMo]"
        if displayTitle:  ax.set_title(title)
        ax.axis("off")

    def __repr__(self):
        """
        Generate a detailed string representation of the image object.
        
        Returns:
            str: Comprehensive description including all image properties
        """
        if not self.colorSpace:
            colorSpaceSTR = "None"
        else: 
            colorSpaceSTR =  self.colorSpace.name
        res =   "<class Image:\n" + \
                "\t name: " + self.name + "\n" + \
                "\t path: " + self.path + "\n" + \
                "\t shape: " + str(self.shape) +  "\n" + \
                "\t type: " + str(self.type) +  "\n" +\
                "\t linear: " + str(self.linear) +  "\n" +\
                "\t scaling to [0,1]: " + str(self.scalingFactor)+"\n" +\
                "\t color space: " + colorSpaceSTR + ">"
        return res

    @staticmethod
    def buildLchColorData(L,c,h,size,width,height):
        """
        Generate synthetic LCH color data for visualization and testing.
        
        Creates artificial color data in LCH space with specified lightness,
        chroma, and hue ranges. Useful for color space visualization and
        testing color transformations.

        Args:
            L (tuple): Lightness range (min, max)
            c (tuple): Chroma range (min, max) 
            h (tuple): Hue range (min, max)
            size (tuple): Output image dimensions (height, width)
            width (str): Primary axis for width ('L', 'c', or 'h')
            height (str): Primary axis for height ('L', 'c', or 'h')
                
        Returns:
            numpy.ndarray: Generated LCH color data with shape (height, width, 3)
        """

        colorData = np.zeros((size[0], size[1],3))
        Lmin, Lmax = L if L[0] < L[1] else (L[1], L[0])
        cmin, cmax = c if c[0] < c[1] else (c[1], c[0])
        hmin, hmax = h

        xmin, xmax = 0, size[1]
        ymin, ymax = 0, size[0]

        if width=='L':
            if height=='c':
                #     +-------------------------------+
                #  c  |                               |
                #     +-------------------------------+
                #                Lightness
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        Lu = Lmin*(1-u)+Lmax*u
                        cv = cmin*(1-v) + cmax*v
                        hh = (hmin+hmax)/2
                        colorData[y,x,:] = [Lu,cv,hh]
            elif height=='h':
                #     +-------------------------------+
                #  h  |                               |
                #     +-------------------------------+
                #                Lightness
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        Lu = Lmin*(1-u)+Lmax*u
                        if hmin <= hmax:
                            hv = hmin*(1-v) + hmax*v
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hv = (hmin-360)*(1-v) + hmax*v
                            if hv < 0: hv = 360 +hv
                        cc = (cmin+cmax)/2
                        colorData[y,x,:] = [Lu,cc,hv]
        elif width=='c':
            if height=='L':
                #     +-------------------------------+
                #  L  |                               |
                #     +-------------------------------+
                #                chroma
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        cu = cmin*(1-u) + cmax*u
                        Lv = Lmin*(1-v)+Lmax*v
                        hh = (hmin+hmax)/2
                        colorData[y,x,:] = [Lv,cu,hh]
            elif height=='h':
                #     +-------------------------------+
                #  h  |                               |
                #     +-------------------------------+
                #                chroma
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        cu = cmin*(1-u) + cmax*u
                        if hmin <= hmax:
                            hv = hmin*(1-v) + hmax*v
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hv = (hmin-360)*(1-v) + hmax*v
                            if hv < 0: hv = 360 +hv
                        LL = (Lmin+Lmax)/2
                        colorData[y,x,:] = [LL,cu,hv]
        elif width == 'h':
            if height=='L':
                #     +-------------------------------+
                #  L  |                               |
                #     +-------------------------------+
                #                hue
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        if hmin <= hmax:
                            hu = hmin*(1-u) + hmax*u
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hu = (hmin-360)*(1-u) + hmax*u
                            if hu < 0: hv = 360 +hu
                        Lv = Lmin*(1-v)+Lmax*v
                        cc = (cmin+cmax)/2
                        colorData[y,x,:] = [Lv,cc,hu]
            elif height=='c':
                #     +-------------------------------+
                #  c  |                               |
                #     +-------------------------------+
                #                hue
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        if hmin <= hmax:
                            hu = hmin*(1-u) + hmax*u
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hu = (hmin-360)*(1-u) + hmax*u
                            if hu < 0: hv = 360 +hu
                        cv = cmin*(1-v)+cmax*v
                        LL = (Lmin+Lmax)/2
                        colorData[y,x,:] = [LL,cv,hu]

        return colorData

    def split(self,widthSegment,heightSegment):
        """
        Divide the image into a grid of sub-images.
        
        Splits the current image into widthSegment × heightSegment sub-images,
        each maintaining the same properties as the original image.

        Args:
            widthSegment (int): Number of horizontal segments
            heightSegment (int): Number of vertical segments

        Returns:
            list: 2D list of Image objects representing the sub-images
                 Arranged as [row][column] where each element is an Image
        """
        imageHeight,imageWidth, _ = self.colorData.shape
        widthLimit = [(i*(imageWidth//widthSegment))  for i in range(widthSegment)]+[imageWidth]
        heightLimit = [(i*(imageHeight//heightSegment))  for i in range(widthSegment)]+[imageHeight]

        res = []

        for line in range(heightSegment):
            lines = []
            for col in range(widthSegment):
                cData =  copy.deepcopy(self.colorData[(heightLimit[line]):(heightLimit[line+1]),(widthLimit[col]):(widthLimit[col+1]),:])
                imgTemp = Image(copy.deepcopy(self.path), copy.deepcopy(self.name), cData, copy.deepcopy(self.type), self.linear, copy.deepcopy(self.colorSpace), self.scalingFactor)
                imgTemp.metadata = copy.deepcopy(self.metadata)
                lines.append(imgTemp)
            res.append(lines)

        return res

    @staticmethod
    def merge(imgList):
        """
        Combine a 2D list of images into a single merged image.
        
        Reconstructs a complete image from sub-images that were previously split,
        assuming all sub-images have compatible dimensions and properties.

        Args:
            imgList (list): 2D list of Image objects to merge

        Returns:
            Image: Single merged image with combined pixel data
        """
        totalWidth= functools.reduce(lambda x,y: x+y, map(lambda img: img.colorData.shape[1],imgList[0]),0)
        totalHeight= functools.reduce(lambda x,y: x+y,map(lambda imgList: imgList[0].shape[0],imgList),0)

        cData = np.zeros((totalHeight,totalWidth,3))

        y = 0
        for line in imgList:
            x=0
            for img in line:
                cData[y:(y+img.colorData.shape[0]),x:(x+img.colorData.shape[1]),:] = img.colorData
                x = x+img.colorData.shape[1]
            y = y + line[-1].colorData.shape[0]
        return Image(copy.deepcopy(imgList[0][0].path), copy.deepcopy(imgList[0][0].name), cData, copy.deepcopy(imgList[0][0].type), imgList[0][0].linear, copy.deepcopy(imgList[0][0].colorSpace), imgList[0][0].scalingFactor)
# -----------------------------------------------------------------------------
# --- Class ColorSpace -------------------------------------------------------
# -----------------------------------------------------------------------------
class ColorSpace(object):
    """
    Color space definitions and factory methods for HDR imaging.
    
    This class provides static methods for creating and managing various color
    spaces used in HDR image processing. It serves as a centralized interface
    for color space creation and configuration.
    """

    @staticmethod
    def Lab():
        """
        Create a Lab color space definition for HDR processing.
        
        Returns:
            colour.RGB_Colourspace: Lab color space configuration
        """
        return colour.RGB_Colourspace('Lab', primaries=np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]), whitepoint=np.array([0.32168, 0.33767]))
     
    @staticmethod                                 
    def Lch():
        """
        Create an LCH color space definition for HDR processing.
        
        Returns:
            colour.RGB_Colourspace: LCH color space configuration
        """
        return colour.RGB_Colourspace('Lch', primaries=np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]), whitepoint=np.array([0.32168, 0.33767]))

    @staticmethod
    def sRGB():
        """
        Create a standard sRGB color space definition.
        
        Returns:
            colour.RGB_Colourspace: sRGB color space configuration
        """
        return colour.models.RGB_COLOURSPACES['sRGB'].copy()

    @staticmethod
    def scRGB():
        """
        Create an extended sRGB color space for HDR content.
        
        This variant of sRGB removes the gamma correction for HDR processing,
        providing linear RGB values with extended range.
        
        Returns:
            colour.RGB_Colourspace: Extended sRGB color space for HDR
        """
        colorSpace = colour.models.RGB_COLOURSPACES['sRGB'].copy()
        colorSpace.cctf_decoding=None
        return colorSpace

    @staticmethod
    def XYZ():
        """
        Create an XYZ color space definition.
        
        Returns:
            colour.RGB_Colourspace: XYZ color space configuration
        """
        return colour.RGB_Colourspace('XYZ', primaries=np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]), whitepoint=np.array([0.32168, 0.33767]))

    @staticmethod
    def build(name='sRGB'):
        """
        Factory method to create color spaces by name.
        
        Args:
            name (str, optional): Color space name ('sRGB', 'scRGB', 'Lab', 'Lch', 'XYZ')
                                 Defaults to 'sRGB'
        
        Returns:
            colour.RGB_Colourspace or None: Requested color space or None if unknown
        """
        cs  = None
        if name== 'sRGB': cs =  ColorSpace.sRGB()
        if name== 'scRGB': cs =  ColorSpace.scRGB()
        if name== 'Lab' : cs =  ColorSpace.Lab()
        if name== 'Lch' : cs =  ColorSpace.Lch()
        if name== 'XYZ' : cs =  ColorSpace.XYZ()
        return cs 
# -----------------------------------------------------------------------------     
# --- Class Histogram --------------------------------------------------------
# -----------------------------------------------------------------------------
class Histogram(object):
    """
    Image histogram computation and analysis for HDR imaging.
    
    This class provides comprehensive histogram generation and analysis capabilities
    for HDR images, supporting both linear and logarithmic scales for different
    dynamic ranges.
    
    Attributes:
        name (str): Descriptive name for the histogram
        channel (channel): Color channel analyzed
        histValue (numpy.ndarray): Histogram bin values
        edgeValue (numpy.ndarray): Histogram bin edges
        logSpace (bool): Whether histogram uses logarithmic spacing
    
    Methods:
        normalise: Normalize histogram values
        plot: Display histogram on matplotlib axis
        toNumpy: Convert histogram to numpy array
        
    Static Methods:
        build: Generate histogram from image and channel
    """
    
    def __init__(self,histValue,edgeValue,name,channel,logSpace=False):
        """
        Initialize a histogram object with computed values.

        Args:
            histValue (numpy.ndarray): Histogram bin counts
            edgeValue (numpy.ndarray): Histogram bin edge positions
            name (str): Descriptive name for the histogram
            channel (channel): Color channel that was analyzed
            logSpace (bool, optional): Whether bins use logarithmic spacing (default: False)
        """
        self.name           = name
        self.channel        = channel
        self.histValue      = histValue
        self.edgeValue      = edgeValue
        self.logSpace       = logSpace    

    def __repr__(self):
        """
        Generate a detailed string representation of the histogram.
        
        Returns:
            str: Comprehensive histogram description
        """
        res =   "<class Histogram: \n" + \
                "t name:"               + self.name                 + "\n"  + \
                "\t nb bins: "        + str(len(self.histValue))    + "\n"  + \
                "\t channel: "        + str(self.channel.name)      + "\n"  + \
                "\t logSpace: "       + str(self.logSpace)          + ">"  
        return res

    def __str__(self):
        """
        Generate a human-readable string representation.
        
        Returns:
            str: Same as __repr__ for consistency
        """
        return self.__repr__()

    def normalise(self,norm=None):
        """
        Normalize histogram values according to specified norm.

        Args:
            norm (str, optional): Normalization method ('probability' or 'dot')
                                 Defaults to 'probability'
                
        Returns:
            Histogram: New normalized histogram object
        """
        res = copy.deepcopy(self)
        if not norm: norm = 'probability'
        if norm == 'probability':
            sum = np.sum(res.histValue)
            res.histValue = res.histValue/sum
        elif norm == 'dot':
            dot2 = np.dot(res.histValue,res.histValue)
            res.histValue = res.histValue/np.sqrt(dot2)
        else:
            print("WARNING[Histogram.normalise(",self.name,"): unknown norm:", norm,"!]")
        return res

    @staticmethod
    def build(img,channel,nbBins=100,range=None,logSpace=None):
        """
        Generate a histogram from an image channel.

        Args:
            img (Image): Source image for histogram generation
            channel (channel): Color channel to analyze
            nbBins (int, optional): Number of histogram bins (default: 100)
            range (tuple, optional): Value range for histogram (min, max)
                                   If None, determined automatically
            logSpace (bool or str, optional): Use logarithmic bin spacing
                                            'auto' determines from image type
                
        Returns:
            Histogram: Generated histogram object with computed values
        """
        # logSpace
        if not isinstance(logSpace,(bool, str)): logSpace = 'auto'
        if isinstance(logSpace,str):
            if logSpace=='auto':
                if img.type == image.imageType.imageType.SDR : logSpace = False
                if img.type == image.imageType.imageType.HDR : logSpace = True
            else: logSpace = False

        channelVector = utils.ndarray2vector(img.getChannel(channel))
        # range
        if not range: 
            if channel.colorSpace() == 'Lab':
                range= (0.0,100.0)
            elif channel.colorSpace() == 'sRGB'or channel.colorSpace() == 'XYZ':
                range= (0.0,1.0)
            else:
                range= (0.0,1.0)

        # compute bins
        if logSpace:
            minChannel = np.amin(channelVector)
            maxChannel = np.amax(channelVector)
            #bins
            bins = 10 ** np.linspace(np.log10(minChannel), np.log10(maxChannel), nbBins+1)
        else:
            bins = np.linspace(range[0],range[1],nbBins+1)

        nphist, npedges = np.histogram(channelVector, bins)

        nphist = nphist/channelVector.shape
        return Histogram(nphist, 
                         npedges, 
                         'hist_'+str(channel)+'_'+img.name, 
                         channel,
                         logSpace = logSpace
                         )

    def plot(self,ax,color='r',shortName=True,title=True):
        """
        Display the histogram on a matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axis for plotting
            color (str, optional): Plot color (default: 'r')
            shortName (bool, optional): Use abbreviated name (default: True)
            title (bool, optional): Show title (default: True)
        """
        if not color : color = 'r'
        ax.plot(self.edgeValue[1:],self.histValue,color)
        if self.logSpace: ax.set_xscale("log")
        name = self.name.split("/")[-1]+"(H("+self.channel.name+"))"if shortName else self.name+"(Histogram:"+self.channel.name+")"
        if title: ax.set_title(name)

    def toNumpy(self):
        """
        Convert histogram values to numpy array.
        
        Returns:
            numpy.ndarray: Histogram bin values as numpy array
        """
        return self.histValue
