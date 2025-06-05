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
# --- Package hdrGUI ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
guiQt.model module: Model classes for uHDR GUI application.

This module implements the Model component of the Model-View-Controller (MVC) 
architecture pattern used in uHDR. It contains data models that manage the 
state and business logic for image gallery, image editing, aesthetics analysis,
and various UI components.

The models handle:
- Image gallery management and pagination
- HDR image processing pipeline parameters
- Tone curve and color editing configurations  
- Geometry transformations and lightness masking
- Aesthetics analysis and color palette extraction
- Threading coordination for parallel processing

Main Classes:
    ImageWidgetModel: Simple image display model
    ImageGalleryModel: Gallery view with pagination and process-pipes
    AppModel: Main application state and directory management
    EditImageModel: Complete HDR editing pipeline model
    ToneCurveModel: B-spline tone curve editing with control points
    LightnessMaskModel: Tone range selection masking
    HDRviewerModel: HDR display configuration
    ImageAestheticsModel: Color palette and aesthetics analysis
    ColorEditorsAutoModel: Automatic color editor configuration

Threading Models:
    AdvanceSliderModel: Slider state management
    LchColorSelectorModel: LCH color space editing
    GeometryModel: Geometric transformations
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------

import os, colour, copy, json, time, sklearn.cluster, math
import pathos.multiprocessing, multiprocessing, functools
import numpy as np
from geomdl import BSpline
from geomdl import utilities

from datetime import datetime

import hdrCore.image, hdrCore.utils, hdrCore.aesthetics, hdrCore.image
from . import controller, thread
import hdrCore.processing, hdrCore.quality
import preferences.preferences as pref

from PyQt5.QtCore import QRunnable

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageWidgetModel(object):
    """
    Simple model for individual image widget display.
    
    Manages the data state for a single image widget that can display
    either numpy arrays or hdrCore.image.Image objects.
    
    Attributes:
        - controller: Reference to the controlling ImageWidgetController
        - image: Image data (numpy.ndarray or hdrCore.image.Image)
    """

    def __init__(self, controller):
        """
        Initialize the image widget model.
        
        Args:
            controller: Parent ImageWidgetController instance
        """
        self.controller = controller
        self.image = None # numpy array or hdrCore.image.Image

    def setImage(self,image): 
        """
        Set the image to be displayed.
        
        Args:
            image (numpy.ndarray or hdrCore.image.Image): Image data to display
        """
        self.image = image

    def getColorData(self): 
        """
        Extract color data from the current image for display.
        
        Returns:
            numpy.ndarray: RGB color data ready for Qt display
        """
        if isinstance(self.image, np.ndarray):
            return self.image
        elif isinstance(self.image, hdrCore.image.Image):
            return self.image.colorData
# ------------------------------------------------------------------------------------------
# --- class ImageGalleryModel --------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageGalleryModel:
    """
    Model for image gallery management with pagination and process-pipes.
    
    Handles a collection of images with their associated processing pipelines,
    supporting paginated display and thumbnail generation. Each image has an
    associated ProcessPipe for HDR editing operations.
    
    Attributes:
        - controller (ImageGalleryController): Parent controller reference
        - imageFilenames (list[str]): List of image file paths
        - processPipes (list[ProcessPipe]): HDR processing pipelines for each image
        - _selectedImage (int): Index of currently selected image (-1 if none)
        - aestheticsModels (list): Aesthetic analysis models for images
    """

    def __init__(self, _controller):
        """
        Initialize the image gallery model.
        
        Args:
            _controller (ImageGalleryController): Parent controller instance
        """
        if pref.verbose:  print(" [MODEL] >> ImageGalleryModel.__init__()")

        self.controller = _controller
        self.imageFilenames = []
        self.processPipes = []
        self._selectedImage= -1

        self.aesthetics = []

    def setSelectedImage(self,id): 
        """
        Set the currently selected image by index.
        
        Args:
            id (int): Index of image to select
        """
        self._selectedImage = id

    def selectedImage(self): 
        """
        Get the index of currently selected image.
        
        Returns:
            int: Selected image index (-1 if none selected)
        """
        return self._selectedImage

    def getSelectedProcessPipe(self):
        """
        Get the ProcessPipe associated with the selected image.
        
        Returns:
            ProcessPipe or None: Processing pipeline for selected image
        """
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.getSelectedProcessPipe(",  ")")

        res = None
        if self._selectedImage != -1: res= self.processPipes[self._selectedImage]
        return res

    def setImages(self, filenames):
        """
        Load a new set of images into the gallery.
        
        Initializes the gallery with new image filenames, resets all process-pipes
        and aesthetics models, then loads the first page.
        
        Args:
            filenames (list[str]): List of image file paths to load
        """
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.setImages(",len(list(copy.deepcopy(filenames))), "images)")

        self.imageFilenames = list(filenames)
        self.imagesMetadata, self.processPipes =  [], [] # reset metadata and processPipes

        self.aestheticsModels = [] # reset aesthetics models

        nbImagePage = controller.GalleryMode.nbRow(self.controller.view.shapeMode)*controller.GalleryMode.nbCol(self.controller.view.shapeMode)
        for f in self.imageFilenames: # load only first page
            self.processPipes.append(None)
        self.controller.updateImages() # update controller to update view
        self.loadPage(0)

    def loadPage(self,nb):
        """
        Load and process images for a specific page.
        
        Loads images for the specified page number, creating ProcessPipes
        for new images and updating existing ones. Uses threading for
        efficient parallel loading.
        
        Args:
            nb (int): Page number to load (0-indexed)
        """
        if pref.verbose:  print(" [MODEL] >> ImageGalleryModel.loadPage(",nb,")")
        nbImagePage = controller.GalleryMode.nbRow(self.controller.view.shapeMode)*controller.GalleryMode.nbCol(self.controller.view.shapeMode)
        min_,max_ = (nb*nbImagePage), ((nb+1)*nbImagePage)

        loadThreads = thread.RequestLoadImage(self)

        for i,f in enumerate(self.imageFilenames[min_:max_]): # load only the current page nb
            if not isinstance(self.processPipes[min_+i],hdrCore.processing.ProcessPipe):
                self.controller.parent.statusBar().showMessage("read image: "+f)
                self.controller.parent.statusBar().repaint()
                loadThreads.requestLoad(min_,i, f)
            else:
                self.controller.view.updateImage(i, self.processPipes[min_+i], f)

    def save(self):
        """
        Save all ProcessPipe configurations to their associated image metadata.
        
        Iterates through all ProcessPipes and saves their parameters as metadata
        in the corresponding image files for persistence.
        """
        if pref.verbose:  print(" [MODEL] >> ImageGalleryModel.save()")

        for i,p in enumerate(self.processPipes):
            if isinstance(p, hdrCore.processing.ProcessPipe): 
                p.getImage().metadata.metadata['processpipe'] = p.toDict()            
                p.getImage().metadata.save()

    def getFilenamesOfCurrentPage(self):
        """
        Get the filenames for images on the current page.
        
        Returns:
            list[str]: Filenames of images currently displayed
        """
        minIdx, maxIdx = self.controller.pageIdx()
        return copy.deepcopy(self.imageFilenames[minIdx:maxIdx])

    def getProcessPipeById(self,i):
        """
        Get ProcessPipe by index.
        
        Args:
            i (int): Index of desired ProcessPipe
            
        Returns:
            ProcessPipe: Processing pipeline at specified index
        """
        return self.processPipes[i]

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AppModel(object):
    """
    Main application model managing global state.
    
    Handles the top-level application state including current working directory,
    image file lists, and HDR display configuration.
    
    Attributes:
        - controller: Reference to parent AppController
        - directory (str): Current working directory path
        - imageFilenames (list[str]): List of image files in directory
        - displayHDRProcess: HDR display process reference
        - displayModel (dict): HDR display configuration with scaling and shape
    """

    def __init__(self, controller):
        """
        Initialize the application model.
        
        Args:
            controller: Parent AppController instance
        """
        # attributes
        self.controller = controller
        self.directory = pref.getImagePath()
        self.imageFilenames = []
        self.displayHDRProcess = None
        #V5
        self.displayModel = {'scaling':12, 'shape':(2160,3840)}

    def setDirectory(self,path):
        """
        Set working directory and scan for supported image files.
        
        Args:
            path (str): Directory path to scan
            
        Returns:
            iterator: Image filenames found in directory
        """
        # read directory and return image filename list
        self.directory =path
        pref.setImagePath(path)
        self.imageFilenames = map(lambda x: os.path.join(self.directory,x),
                                  hdrCore.utils.filterlistdir(self.directory,('.jpg','.JPG','.hdr','.HDR')))

        return self.imageFilenames
# ------------------------------------------------------------------------------------------
# --- Class EditImageModel -----------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class EditImageModel(object):
    """
    Model for comprehensive HDR image editing pipeline.
    
    Manages the complete HDR image editing workflow including exposure adjustment,
    contrast control, tone curve manipulation, color editing, and geometric 
    transformations. Coordinates with threading system for real-time preview.
    
    The model builds and manages a ProcessPipe containing:
    - Exposure adjustment
    - Contrast control
    - B-spline tone curve editing
    - Lightness masking for selective editing
    - Saturation adjustment
    - Five independent color editors for targeted color correction
    - Geometric transformations (rotation, cropping)
    
    Attributes:
        - controller: Reference to EditImageController
        - autoPreviewHDR (bool): Enable automatic HDR preview updates
        - processpipe (ProcessPipe): Current HDR processing pipeline
        - requestCompute (RequestCompute): Threading coordinator for real-time updates
    """
    def __init__(self,controller):
        """
        Initialize the HDR image editing model.
        
        Args:
            controller: Parent EditImageController instance
        """
        self.controller = controller

        self.autoPreviewHDR = False

        # ref to ImageGalleryModel.processPipes[ImageGalleryModel._selectedImage]
        self.processpipe = None 

        # create a RequestCompute
        self.requestCompute  = thread.RequestCompute(self)

    def getProcessPipe(self): 
        """
        Get the current processing pipeline.
        
        Returns:
            ProcessPipe: Current HDR processing pipeline
        """
        return self.processpipe

    def setProcessPipe(self, processPipe): 
        """
        Set a new processing pipeline and trigger computation.
        
        Args:
            processPipe (ProcessPipe): New processing pipeline to use
            
        Returns:
            bool: True if pipeline was set successfully, False if system busy
        """
        if self.requestCompute.readyToRun:
            self.processpipe = processPipe
            self.requestCompute.setProcessPipe(self.processpipe)
            self.processpipe.compute()
            if self.controller.previewHDR and self.autoPreviewHDR:
                img = self.processpipe.getImage(toneMap = False)
                self.controller.controllerHDR.displayIMG(img)
            return True
        else:
            return False

    @staticmethod
    def buildProcessPipe():
        """
        Create a default HDR processing pipeline.
        
        Builds a complete ProcessPipe with all editing stages configured
        with default parameters. The pipeline includes exposure, contrast,
        tone curve, masking, saturation, color editors, and geometry.
        
        WARNING: The initial pipe does not have an input image and must be
        configured with setImage() before use.
        
        Returns:
            ProcessPipe: Configured processing pipeline with default parameters
        """
        processPipe = hdrCore.processing.ProcessPipe()

        # exposure ---------------------------------------------------------------------------------------------------------
        defaultParameterEV = {'EV': 0}                                              
        idExposureProcessNode = processPipe.append(hdrCore.processing.exposure(), paramDict=None,name="exposure")   
        processPipe.setParameters(idExposureProcessNode, defaultParameterEV)                                        

        # contrast ---------------------------------------------------------------------------------------------------------
        defaultParameterContrast = {'contrast': 0}                                  
        idContrastProcessNode = processPipe.append(hdrCore.processing.contrast(), paramDict=None,  name="contrast") 
        processPipe.setParameters(idContrastProcessNode, defaultParameterContrast)                                  

        #tonecurve ---------------------------------------------------------------------------------------------------------
        defaultParameterYcurve = {'start':[0,0], 
                                  'shadows': [10,10],
                                  'blacks': [30,30], 
                                  'mediums': [50,50], 
                                  'whites': [70,70], 
                                  'highlights': [90,90], 
                                  'end': [100,100]}                         
        idYcurveProcessNode = processPipe.append(hdrCore.processing.Ycurve(), paramDict=None,name="tonecurve")      
        processPipe.setParameters(idYcurveProcessNode, defaultParameterYcurve)   
        
        # masklightness ---------------------------------------------------------------------------------------------------------
        defaultMask = { 'shadows': False, 
                       'blacks': False, 
                       'mediums': False, 
                       'whites': False, 
                       'highlights': False}
        idLightnessMaskProcessNode = processPipe.append(hdrCore.processing.lightnessMask(), paramDict=None, name="lightnessmask")  
        processPipe.setParameters(idLightnessMaskProcessNode, defaultMask)  

        # saturation ---------------------------------------------------------------------------------------------------------
        defaultValue = {'saturation': 0.0,  'method': 'gamma'}
        idSaturationProcessNode = processPipe.append(hdrCore.processing.saturation(), paramDict=None, name="saturation")    
        processPipe.setParameters(idSaturationProcessNode, defaultValue)                     

        # colorEditor0 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor0= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor0ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor0")  
        processPipe.setParameters(idColorEditor0ProcessNode, defaultParameterColorEditor0)

        # colorEditor1 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor1= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor1ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor1")  
        processPipe.setParameters(idColorEditor1ProcessNode, defaultParameterColorEditor1)
        
        # colorEditor2 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor2= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor2ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor2")  
        processPipe.setParameters(idColorEditor2ProcessNode, defaultParameterColorEditor2)
        
        # colorEditor3 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor3= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor3ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor3")  
        processPipe.setParameters(idColorEditor3ProcessNode, defaultParameterColorEditor3)
        
        # colorEditor4 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor4= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor4ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor4")  
        processPipe.setParameters(idColorEditor4ProcessNode, defaultParameterColorEditor4)

        # geometry ---------------------------------------------------------------------------------------------------------
        defaultValue = { 'ratio': (16,9), 'up': 0,'rotation': 0.0}
        idGeometryNode = processPipe.append(hdrCore.processing.geometry(), paramDict=None, name="geometry")    
        processPipe.setParameters(idGeometryNode, defaultValue)
        # ------------ --------------------------------------------------------------------------------------------------------- 

        return processPipe

    def autoExposure(self):
        """
        Automatically calculate and apply optimal exposure settings.
        
        Uses the exposure process's automatic calculation method to determine
        optimal exposure value based on image content and applies it to the pipeline.
        
        Returns:
            numpy.ndarray: Processed image with auto-exposure applied
        """
        if pref.verbose:  print(" [MODEL] >> EditImageModel.autoExposure(",")")

        id = self.processpipe.getProcessNodeByName("exposure")
        exposureProcess = self.processpipe.processNodes[id].process
        img= self.processpipe.getInputImage()
        EV = exposureProcess.auto(img)
        self.processpipe.setParameters(id,EV)

        self.processpipe.compute()

        if self.controller.previewHDR and self.autoPreviewHDR:
            img = self.processpipe.getImage(toneMap = False)
            self.controller.controllerHDR.displayIMG(img)

        return self.processpipe.getImage()

    def changeExposure(self,value):
        """
        Manually adjust exposure value.
        
        Args:
            value (float): Exposure adjustment in EV stops
        """
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeExposure(",value,")")

        id = self.processpipe.getProcessNodeByName("exposure")
        self.requestCompute.requestCompute(id,{'EV': value})

    def getEV(self):
        """
        Get current exposure value settings.
        
        Returns:
            dict: Current exposure parameters
        """
        if pref.verbose:  print(" [MODEL] >> EditImageModel.getEV(",")")
        id = self.processpipe.getProcessNodeByName("exposure")
        return self.processpipe.getParameters(id)

    def changeContrast(self,value):
        """
        Adjust contrast parameters.
        
        Args:
            value (float): Contrast adjustment value
        """
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeContrast(",value,")")

        id = self.processpipe.getProcessNodeByName("contrast")
        self.requestCompute.requestCompute(id,{'contrast': value})

    def changeToneCurve(self,controlPoints):
        """
        Update tone curve control points.
        
        Args:
            controlPoints (dict): B-spline control point dictionary
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeToneCurve(",")")

        id = self.processpipe.getProcessNodeByName("tonecurve")
        self.requestCompute.requestCompute(id,controlPoints)

    def changeLightnessMask(self, maskValues):
        """
        Modify lightness masking parameters.
        
        Args:
            maskValues (dict): Mask configuration for different tone ranges
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeLightnessMask(",maskValues,")")

        id = self.processpipe.getProcessNodeByName("lightnessmask")
        self.requestCompute.requestCompute(id,maskValues)

    def changeSaturation(self,value):
        """
        Adjust global saturation.
        
        Args:
            value (float): Saturation adjustment value
        """
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeSaturation(",value,")")

        id = self.processpipe.getProcessNodeByName("saturation")
        self.requestCompute.requestCompute(id,{'saturation': value, 'method': 'gamma'})

    def changeColorEditor(self, values, idName):
        """
        Update color editor parameters.
        
        Args:
            values (dict): Color editor configuration
            idName (str): Name of color editor process node
        """
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeColorEditor(",values,")")

        id = self.processpipe.getProcessNodeByName(idName)
        self.requestCompute.requestCompute(id,values)

    def changeGeometry(self, values):
        """
        Modify geometric transformation parameters.
        
        Args:
            values (dict): Geometry configuration including rotation and cropping
        """
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeGeometry(",values,")")

        id = self.processpipe.getProcessNodeByName("geometry")
        self.requestCompute.requestCompute(id,values)

    def updateImage(self, imgTM):
        """
        Update view with newly processed image.
        
        Args:
            imgTM (numpy.ndarray): Tone-mapped image for display
        """
        self.controller.updateImage(imgTM)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceSliderModel():
    """
    Simple model for advanced slider controls.
    
    Manages state for slider widgets with additional functionality like
    auto adjustment and value persistence.
    
    Attributes:
        - controller: Reference to parent controller
        - value (float): Current slider value
    """
    def __init__(self, controller, value):
        """
        Initialize advanced slider model.
        
        Args:
            controller: Parent controller instance
            value (float): Initial slider value
        """
        self.controller = controller
        self.value = value
    def setValue(self, value): 
        """
        Set new slider value.
        
        Args:
            value (float): New slider value
        """
        self.value =  value
    def toDict(self): 
        """
        Export slider state as dictionary.
        
        Returns:
            dict: Slider state with 'value' key
        """
        return {'value': self.value}
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ToneCurveModel():
    """
    Model for B-spline tone curve editing with multiple control points.
    
    Manages a tone curve represented as a B-spline with seven control points:
    start, shadows, blacks, mediums, whites, highlights, and end. Provides
    automatic scaling and constraint validation to maintain curve monotonicity.
    
    The tone curve operates on a coordinate system where:
    - X-axis: Input luminance (0-100)
    - Y-axis: Output luminance (0-100) 
    - Control points define the curve shape
    
    Grid representation:
        +-------+-------+-------+-------+-------+                             [o]
        |       |       |       |       |       |                              ^
        |       |       |       |       |   o   |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |       |       |       |   o   |       |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |       |       |   o   |       |       |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |       |   o   |       |       |       |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |   o   |       |       |       |       |                              |
        |       |       |       |       |       |                              |
       [o]-------+-------+-------+-------+-------+-----------------------------+ 
  zeros ^ shadows  black   medium  white  highlights                          200

    Attributes:
        - control (dict): Current control point positions
        - default (dict): Default control point positions for reset
        - curve (BSpline.Curve): B-spline curve object
        - points (numpy.ndarray): Evaluated curve points
    """
    def __init__(self):
        """
        Initialize tone curve model with default control points.
        """
        if pref.verbose: print(" [MODEL] >> ToneCourveModel.__init__()")

        #  start
        self.control = {'start':[0.0,0.0], 'shadows': [10.0,10.0], 'blacks': [30.0,30.0], 'mediums': [50.0,50.0], 'whites': [70.0,70.0], 'highlights': [90.0,90.0], 'end': [100.0,100.0]}
        self.default = {'start':[0.0,0.0], 'shadows': [10.0,10.0], 'blacks': [30.0,30.0], 'mediums': [50.0,50.0], 'whites': [70.0,70.0], 'highlights': [90.0,90.0], 'end': [100.0,100.0]}

        self.curve =    BSpline.Curve()
        self.curve.degree = 2
        self.points =None

    def evaluate(self):
        """
        Evaluate B-spline curve from current control points.
        
        Constructs a degree-2 B-spline curve using the current control points,
        generates appropriate knot vector, and evaluates the curve.
        
        Returns:
            numpy.ndarray: Array of (x,y) points representing the evaluated curve
        """
        if pref.verbose: print(" [MODEL] >> ToneCurveModel.evaluate(",")")

        #self.curve.ctrlpts = copy.deepcopy([self.control['start'],self.control['shadows'],self.control['blacks'],self.control['mediums'], self.control['whites'], self.control['highlights'], self.control['end']])
        self.curve.ctrlpts = copy.deepcopy([self.control['start'],self.control['shadows'],self.control['blacks'],self.control['mediums'], self.control['whites'], self.control['highlights'], [200, self.control['end'][1]]])
        # auto-generate knot vector
        self.curve.knotvector = utilities.generate_knot_vector(self.curve.degree, len(self.curve.ctrlpts))
        # evaluate curve and get points
        self.points = np.asarray(self.curve.evalpts)

        return self.points

    def setValue(self, key, value, autoScale=False):
        """
        Update a control point value with constraints and optional auto-scaling.
        
        Updates the specified control point while maintaining curve monotonicity.
        Optionally performs auto-scaling of adjacent points to maintain smooth
        transitions when constraints would be violated.
        
        Args:
            key (str): Control point name ('shadows', 'blacks', etc.)
            value (int): New Y-coordinate value for the control point
            autoScale (bool): Enable automatic scaling of adjacent points
            
        Returns:
            dict: Updated control points dictionary
        """
        if pref.verbose: print(" [MODEL] >> ToneCurveModel.setValue(",key,", ",value,", autoScale=",autoScale,")")
        value = int(value)
        # check key
        if key in self.control.keys():
            # transform in list
            listKeys = list(self.control.keys())
            listValues = np.asarray(list(self.control.values()))
            index = listKeys.index(key)

            if (listValues[:index,1] <= value).all() and (value <= listValues[index+1:,1]).all():
                # can change
                oldValue = self.control[listKeys[index]]
                self.control[listKeys[index]] = [oldValue[0],value]            
            elif not (value <= listValues[index+1:,1]).all():
                if autoScale:
                    minValue = min(listValues[index:,1])
                    maxValue = listValues[-1:,1]
                    u = (listValues[index+1:,1] -minValue)/(maxValue-minValue)

                    newValues = value*(1- u)+ u*maxValue
                    for i,v in enumerate(newValues):
                        oldValue = self.control[listKeys[i+index+1]]
                        self.control[listKeys[i+index+1]] = [oldValue[0],np.round(v)]
                else:
                    # not autoScale, set to minValue
                    oldValue = self.control[listKeys[index]]
                    minValue = min(listValues[index+1:,1])
                    self.control[listKeys[index]] = [oldValue[0],minValue]
            elif not (listValues[:index,1] <= value).all():
                if autoScale:
                    minValue = listValues[0,1]
                    maxValue = max(listValues[:index,1])
                    u = (listValues[:index,1] -minValue)/(maxValue-minValue)
                    newValues = minValue*(1- u)+ u*value
                    for i,v in enumerate(newValues):
                        oldValue = self.control[listKeys[i]]
                        self.control[listKeys[i]] = [oldValue[0],np.round(v)]
                else:
                    # not autoScale, set to maxValue
                    oldValue = self.control[listKeys[index]]
                    maxValue =  max(listValues[:index,1])
                    self.control[listKeys[index]] = [oldValue[0],maxValue]

        return self.control

    def setValues(self, controlPointsDict):
        """
        Set all control points from dictionary.
        
        Args:
            controlPointsDict (dict): Dictionary of control point coordinates
        """
        self.control = copy.deepcopy(controlPointsDict)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class LightnessMaskModel():
    """
    Model for lightness-based masking to enable selective tone range editing.
    
    Manages boolean masks for different luminance ranges allowing selective
    application of adjustments to specific tonal areas of the image.
    
    Tone ranges:
    - shadows: Darkest areas
    - blacks: Dark areas 
    - mediums: Mid-tones
    - whites: Bright areas
    - highlights: Brightest areas
    
    Attributes:
        - controller: Reference to parent controller
        - masks (dict): Boolean mask state for each tone range
    """
    def __init__(self, _controller):
        """
        Initialize lightness mask model.
        
        Args:
            _controller: Parent LightnessMaskController instance
        """
        self.controller = _controller

        self.masks = {'shadows': False, 'blacks':False, 'mediums':False, 'whites':False, 'highlights':False}

    def maskChange(self, key, on_off):
        """
        Toggle mask state for a specific tone range.
        
        Args:
            key (str): Tone range name ('shadows', 'blacks', etc.)
            on_off (bool): New mask state
            
        Returns:
            dict: Updated mask dictionary
        """
        if key in self.masks.keys(): self.masks[key] = on_off
        return copy.deepcopy(self.masks)

    def setValues(self, values): 
        """
        Set all mask values from dictionary.
        
        Args:
            values (dict): New mask configuration
        """
        self.masks = copy.deepcopy(values)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageInfoModel(object):
    """
    Model for image information and metadata management.
    
    Handles display and editing of image metadata including EXIF data
    and user-defined tags. Supports modification of custom metadata
    fields for workflow management.
    
    Attributes:
        - controller: Reference to parent controller
        - processPipe (ProcessPipe): Current image processing pipeline
    """

    def __init__(self,controller):
        """
        Initialize image info model.
        
        Args:
            controller: Parent ImageInfoController instance
        """
        self.controller = controller

        # ref to ImageGalleryModel.processPipes[ImageGalleryModel._selectedImage]
        self.processPipe = None 

    def getProcessPipe(self): 
        """
        Get current processing pipeline.
        
        Returns:
            ProcessPipe or None: Current processing pipeline
        """
        return self.processPipe

    def setProcessPipe(self, processPipe):  
        """
        Set new processing pipeline.
        
        Args:
            processPipe (ProcessPipe): New processing pipeline
        """
        self.processPipe = processPipe

    def changeMeta(self,tagGroup,tag, on_off): 
        """
        Update metadata field value.
        
        Modifies user-defined metadata tags within the processing pipeline
        and updates the metadata structure accordingly.
        
        Args:
            tagGroup (str): Metadata group name
            tag (str): Specific tag name within group
            on_off (bool): New tag value
        """
        if pref.verbose:  print(" [MODEL] >> ImageInfoModel.changeMeta(",tagGroup,",",tag,",", on_off,")")

        if isinstance(self.processPipe, hdrCore.processing.ProcessPipe):
            tagRootName = self.processPipe.getImage().metadata.otherTags.getTagsRootName()
            tags = copy.deepcopy(self.processPipe.getImage().metadata.metadata[tagRootName])
            found, updatedMeta = False, []
            for tt in tags:
                if tagGroup in tt.keys():
                    if tag in tt[tagGroup].keys():
                        found = True
                        tt[tagGroup][tag] = on_off 
                updatedMeta.append(copy.deepcopy(tt))
            self.processPipe.updateUserMeta(tagRootName,updatedMeta)
            if pref.verbose: 
                print(" [MODEL] >> ImageInfoModel.changeUseCase(",")")
                for tt in updatedMeta: print(tt)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class CurveControlModel(object): 
    """
    Base model class for curve control interfaces.
    
    Placeholder model for potential curve control functionality.
    Currently serves as a base class or interface definition.
    """
    pass
# ------------------------------------------------------------------------------------------
# ---- HDRviewerModel ----------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class HDRviewerModel(object):
    """
    Model for HDR display configuration and management.
    
    Manages HDR display settings including scaling and display dimensions
    for external HDR monitor output. Handles configuration persistence
    and current image reference for HDR preview.
    
    Attributes:
        - controller: Reference to parent HDRviewerController
        - currentIMG: Currently displayed HDR image
        - displayModel (dict): HDR display configuration with scaling and shape
    """
    def __init__(self,_controller):
        """
        Initialize HDR viewer model.
        
        Args:
            _controller: Parent HDRviewerController instance
        """
        if pref.verbose: print(" [MODEL] >> HDRviewerModel.__init__(",")")

        self.controller = _controller

        # current image
        self.currentIMG = None

        self.displayModel = pref.getHDRdisplay()

    def scaling(self): 
        """
        Get current HDR display scaling factor.
        
        Returns:
            float: Scaling factor for HDR display
        """
        if pref.verbose: print(f" [MODEL] >> HDRviewerModel.scaling():{self.displayModel['scaling']}")
        return self.displayModel['scaling']

    def shape(self): 
        """
        Get current HDR display dimensions.
        
        Returns:
            tuple: (height, width) of HDR display
        """
        if pref.verbose: print(f" [MODEL] >> HDRviewerModel.shape():{self.displayModel['shape']}")        
        return self.displayModel['shape']
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class LchColorSelectorModel(object):
    """
    Model for LCH color space selection and editing.
    
    Manages color selection in LCH (Lightness, Chroma, Hue) color space
    with independent range selection for each component and editing
    parameters for targeted color adjustments.
    
    LCH Color Space Components:
    - Lightness: 0-100 (brightness)
    - Chroma: 0-100 (saturation/colorfulness)  
    - Hue: 0-360 (color angle)
    
    Attributes:
        - controller: Reference to parent controller
        - lightnessSelection (tuple): (min, max) lightness range
        - chromaSelection (tuple): (min, max) chroma range
        - hueSelection (tuple): (min, max) hue range
        - exposure (float): Exposure adjustment for selected colors
        - hueShift (float): Hue shift for selected colors
        - contrast (float): Contrast adjustment for selected colors
        - saturation (float): Saturation adjustment for selected colors
        - mask (bool): Display selection mask
        - default (dict): Default values for reset
    """
    def __init__(self, _controller):
        """
        Initialize LCH color selector model.
        
        Args:
            _controller: Parent LchColorSelectorController instance
        """
        self.controller = _controller

        self.lightnessSelection =   (0,100)     # min, max
        self.chromaSelection =      (0,100)     # min, max
        self.hueSelection =         (0,360)     # min, max

        self.exposure =     0.0
        self.hueShift =     0.0
        self.contrast =     0.0
        self.saturation =   0.0

        self.mask =         False

        # -------------
        self.default = {
            "selection": {"lightness": [ 0, 100 ],"chroma": [ 0, 100 ],"hue": [ 0, 360 ]},
            "edit": {"hue": 0,"exposure": 0,"contrast": 0,"saturation": 0},
            "mask": False}
        # -------------


    def setHueSelection(self, hMin,hMax):
        """
        Set hue range selection.
        
        Args:
            hMin (float): Minimum hue value (0-360)
            hMax (float): Maximum hue value (0-360)
            
        Returns:
            dict: Updated complete configuration
        """
        self.hueSelection = hMin,hMax
        return self.getValues()

    def setChromaSelection(self, cMin, cMax):              
        """
        Set chroma range selection.
        
        Args:
            cMin (float): Minimum chroma value (0-100)
            cMax (float): Maximum chroma value (0-100)
            
        Returns:
            dict: Updated complete configuration
        """
        self.chromaSelection = cMin, cMax
        return self.getValues()

    def setLightnessSelection(self, lMin, lMax):              
        """
        Set lightness range selection.
        
        Args:
            lMin (float): Minimum lightness value (0-100)
            lMax (float): Maximum lightness value (0-100)
            
        Returns:
            dict: Updated complete configuration
        """
        self.lightnessSelection = lMin, lMax
        return self.getValues()

    def setExposure(self,ev):
        """
        Set exposure adjustment for selected colors.
        
        Args:
            ev (float): Exposure adjustment in stops
            
        Returns:
            dict: Updated complete configuration
        """
        self.exposure = ev
        return self.getValues()

    def setHueShift(self,hs):
        """
        Set hue shift amount for selected colors.
        
        Args:
            hs (float): Hue shift in degrees (-180 to +180)
            
        Returns:
            dict: Updated complete configuration
        """
        self.hueShift = hs
        return self.getValues()

    def setContrast(self, contrast):
        """
        Set contrast adjustment for selected colors.
        
        Args:
            contrast (float): Contrast adjustment value
            
        Returns:
            dict: Updated complete configuration
        """
        self.contrast = contrast
        return self.getValues()

    def setSaturation(self, saturation): 
        """
        Set saturation adjustment for selected colors.
        
        Args:
            saturation (float): Saturation adjustment value
            
        Returns:
            dict: Updated complete configuration
        """
        self.saturation = saturation
        return self.getValues()

    def setMask(self, value): 
        """
        Toggle selection mask display.
        
        Args:
            value (bool): Enable/disable mask display
            
        Returns:
            dict: Updated complete configuration
        """
        self.mask = value
        return self.getValues()

    def getValues(self):
        """
        Get complete LCH color selector configuration.
        
        Returns:
            dict: Complete configuration with selection ranges and edit parameters
        """
        return {
            'selection':    {'lightness': self.lightnessSelection ,'chroma': self.chromaSelection,'hue':self.hueSelection}, 
            'edit':         {'hue':self.hueShift,'exposure':self.exposure,'contrast':self.contrast,'saturation':self.saturation}, 
            'mask':         self.mask
            }

    def setValues(self, values):
        """
        Set configuration from dictionary with safe defaults.
        
        Args:
            values (dict): Configuration dictionary with selection and edit parameters
        """
        self.lightnessSelection =   values['selection']['lightness']    if 'lightness' in values['selection'].keys() else (0,100)
        self.chromaSelection =      values['selection']['chroma']       if 'chroma' in values['selection'].keys() else (0,100)
        self.hueSelection =         values['selection']['hue']          if 'hue' in values['selection'].keys() else (0,360)

        self.exposure =     values['edit']['exposure']      if 'exposure' in values['edit'].keys() else 0
        self.hueShift =     values['edit']['hue']           if 'hue' in values['edit'].keys() else 0
        self.contrast =     values['edit']['contrast']      if 'contrast' in values['edit'].keys() else 0
        self.saturation =   values['edit']['saturation']    if 'saturation' in values['edit'].keys() else 0

        self.mask =         values['mask']  if 'mask' in values.keys() else False
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class GeometryModel(object):
    """
    Model for geometric transformations including rotation and cropping.
    
    Manages geometric transformation parameters for image processing
    including aspect ratio cropping and rotation adjustments.
    
    Attributes:
        - controller: Reference to parent controller
        - ratio (tuple): Aspect ratio for cropping (width, height)
        - up (float): Vertical cropping adjustment
        - rotation (float): Rotation angle in degrees
    """
    def __init__(self, _controller):
        """
        Initialize geometry model.
        
        Args:
            _controller: Parent GeometryController instance
        """
        self.controller = _controller

        self.ratio =    (16,9)    
        self.up =       0.0
        self.rotation = 0.0 

    def setCroppingVerticalAdjustement(self,up):
        """
        Set vertical cropping adjustment.
        
        Args:
            up (float): Vertical offset for cropping
            
        Returns:
            dict: Updated geometry configuration
        """
        self.up = up
        return self.getValues()

    def setRotation(self, rotation):              
        """
        Set rotation angle.
        
        Args:
            rotation (float): Rotation angle in degrees
            
        Returns:
            dict: Updated geometry configuration
        """
        self.rotation = rotation
        return self.getValues()

    def getValues(self):
        """
        Get complete geometry configuration.
        
        Returns:
            dict: Geometry parameters including ratio, vertical adjustment, and rotation
        """
        return { 'ratio': self.ratio, 'up': self.up,'rotation': self.rotation}

    def setValues(self, values):
        """
        Set geometry configuration from dictionary.
        
        Args:
            values (dict): Geometry configuration with ratio, up, and rotation
        """
        self.ratio =    values['ratio']     if 'ratio'      in values.keys() else (16,9)
        self.up =       values['up']        if 'up'         in values.keys() else 0.0
        self.rotation = values['rotation']  if 'rotation'   in values.keys() else 0.0
# ------------------------------------------------------------------------------------------
# ---- Class AestheticsImageModel ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageAestheticsModel:
    """
    Model for image aesthetics analysis and color palette extraction.
    
    Manages aesthetic analysis of HDR images including color palette generation,
    composition analysis, and visual quality assessment. Uses K-means clustering
    for dominant color extraction and provides visual representations.
    
    Attributes:
        - parent (ImageAestheticsController): Parent controller reference
        - requireUpdate (bool): Flag indicating analysis needs refresh
        - processPipe (ProcessPipe): Current image processing pipeline
        - colorPalette (Palette): Extracted color palette from image
    """
    def __init__(self, parent):
        """
        Initialize image aesthetics model.
        
        Args:
            parent (ImageAestheticsController): Parent controller instance
        """
        if pref.verbose: print(" [MODEL] >> ImageAestheticsModel.__init__(",")")
        
        self.parent = parent

        # processPipeHasChanged
        self.requireUpdate = True


        # ref to ImageGalleryModel.processPipes[ImageGalleryModel._selectedImage]
        self.processPipe = None 

        # color palette
        self.colorPalette = hdrCore.aesthetics.Palette('defaultLab5',
                                                       np.linspace([0,0,0],[100,0,0],5),
                                                       hdrCore.image.ColorSpace.build('Lab'), 
                                                       hdrCore.image.imageType.SDR)
    # ------------------------------------------------------------------------------------------
    def getProcessPipe(self): 
        """
        Get current processing pipeline.
        
        Returns:
            ProcessPipe or None: Current processing pipeline
        """
        return self.processPipe
    # ------------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe):  
        """
        Set new processing pipeline and trigger aesthetics analysis.
        
        If the pipeline has changed, triggers color palette extraction
        and aesthetic analysis using K-means clustering on the image data.
        
        Args:
            processPipe (ProcessPipe): New processing pipeline to analyze
        """
        if pref.verbose: print(" [MODEL] >> ImageAestheticsModel.setProcessPipe(",")")

        if processPipe != self.processPipe:
        
            self.processPipe = processPipe
            self.requireUpdate = True

        if self.requireUpdate:

            self.colorPalette = hdrCore.aesthetics.Palette.build(self.processPipe)
            # COMPUTE IMAGE OF PALETTE
            paletteIMG = self.colorPalette.createImageOfPalette()

            self.endComputing()

        else: pass
    # ------------------------------------------------------------------------------------------
    def endComputing(self):
        """
        Mark aesthetics analysis as complete.
        
        Resets the requireUpdate flag to indicate that analysis is current.
        """
        self.requireUpdate = False
    # ------------------------------------------------------------------------------------------
    def getPaletteImage(self):
        """
        Get visual representation of the extracted color palette.
        
        Returns:
            numpy.ndarray: RGB image showing the color palette as colored bars
        """
        return self.colorPalette.createImageOfPalette()
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ColorEditorsAutoModel:
    """
    Model for automatic color editor configuration using K-means clustering.
    
    Automatically analyzes an image to extract dominant colors and configures
    multiple color editors with appropriate LCH selection ranges for targeted
    color correction. Uses K-means clustering in Lab color space.
    
    Attributes:
        - controller: Reference to parent controller
        - processStepId (str): Name of processing step to analyze
        - nbColors (int): Number of dominant colors to extract
        - removeBlack (bool): Whether to exclude dark/black regions from analysis
    """
    def __init__(self,_controller, processStepName,nbColors, removeBlack= True):
        """
        Initialize automatic color editors model.
        
        Args:
            _controller: Parent ColorEditorsAutoController instance
            processStepName (str): Name of processing step to analyze
            nbColors (int): Number of dominant colors to extract (2-8)
            removeBlack (bool): Exclude dark regions from analysis
        """
        self.controller = _controller
        self.processStepId = processStepName
        self.nbColors = nbColors
        self.removeBlack = removeBlack

    def compute(self):
        """
        Compute automatic color editor configurations using K-means clustering.
        
        Analyzes the image at the specified processing step using K-means clustering
        in Lab color space to identify dominant colors. Generates LCH selection
        ranges for each color editor with appropriate hue, chroma, and lightness
        boundaries.
        
        Algorithm:
        1. Extract image data from specified processing step
        2. Convert to Lab color space if needed
        3. Apply K-means clustering with (nbColors + 1) clusters
        4. Remove darkest cluster if removeBlack is enabled
        5. Convert cluster centers to LCH space
        6. Sort by hue angle
        7. Generate selection ranges with ±25 tolerance for each cluster
        
        Returns:
            list[dict]: List of color editor configurations with selection and edit parameters
        """
        # get image according to processId
        processPipe = self.controller.parent.controller.getProcessPipe()
        if processPipe != None:
            image_ = processPipe.processNodes[processPipe.getProcessNodeByName(self.processStepId)].outputImage

            if image_.colorSpace.name == 'Lch':
                LchPixels = image_.colorData
            elif image_.colorSpace.name == 'sRGB':
                if image_.linear: 
                    colorLab = hdrCore.processing.sRGB_to_Lab(image_.colorData, apply_cctf_decoding=False)
                    LchPixels = colour.Lab_to_LCHab(colorLab)
                else:
                    colorLab = hdrCore.processing.sRGB_to_Lab(image_.colorData, apply_cctf_decoding=True)
                    LchPixels = colour.Lab_to_LCHab(colorLab)

            # to Lab then to Vector
            LabPixels = colour.LCHab_to_Lab(LchPixels)
            LabPixelsVector = hdrCore.utils.ndarray2vector(LabPixels)

            # k-means: nb cluster = nbColors + 1
            kmeans_cluster_Lab = sklearn.cluster.KMeans(n_clusters=self.nbColors+1)
            kmeans_cluster_Lab.fit(LabPixelsVector)
            cluster_centers_Lab = kmeans_cluster_Lab.cluster_centers_
            cluster_labels_Lab = kmeans_cluster_Lab.labels_
                
            # remove darkness one
            idxLmin = np.argmin(cluster_centers_Lab[:,0])                           # idx of darkness
            cluster_centers_Lab = np.delete(cluster_centers_Lab, idxLmin, axis=0)   # remove min from cluster_centers_Lab

            # go to Lch
            cluster_centers_Lch = colour.Lab_to_LCHab(cluster_centers_Lab) 

            # sort cluster by hue
            cluster_centersIdx = np.argsort(cluster_centers_Lch[:,2])

            dictValuesList = []
            for j in range(len(cluster_centersIdx)):
                i = cluster_centersIdx[j]
                if j==0: Hmin = 0
                else: Hmin = 0.5*(cluster_centers_Lch[cluster_centersIdx[j-1]][2] + cluster_centers_Lch[cluster_centersIdx[j]][2])
                if j == len(cluster_centersIdx)-1: Hmax = 360
                else: Hmax = 0.5*(cluster_centers_Lch[cluster_centersIdx[j]][2] + cluster_centers_Lch[cluster_centersIdx[j+1]][2])

                Cmin = max(0, cluster_centers_Lch[cluster_centersIdx[j]][1]-25) 
                Cmax = min(100, cluster_centers_Lch[cluster_centersIdx[j]][1]+25) 

                Lmin = max(0, cluster_centers_Lch[cluster_centersIdx[j]][0]-25) 
                Lmax = min(100, cluster_centers_Lch[cluster_centersIdx[j]][0]+25) 

                dictSegment =  {
                    "selection": {
                        "lightness":    [ int(Lmin),  int(Lmax)],
                        "chroma":       [ int(Cmin),  int(Cmax)],
                        "hue":          [ int(Hmin),  int(Hmax)]},
                    "edit": {"hue": 0,"exposure": 0,"contrast": 0,"saturation": 0},
                    "mask": False}
                dictValuesList.append(dictSegment)

            return dictValuesList






