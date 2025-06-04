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
uHDR GUI Controller Module

This module implements the Model-View-Controller (MVC) pattern for the uHDR
graphical user interface. It provides comprehensive controller classes that
manage user interactions, coordinate between GUI components, and handle
image processing workflows.

The controller system manages multiple aspects of the HDR editing interface:
- Image gallery navigation and display
- HDR image editing controls and parameters
- Processing pipeline management
- File operations and metadata handling
- Multi-monitor HDR display support

Classes:
    GalleryMode: Enumeration for gallery display layouts
    ImageWidgetController: Individual image widget management
    ImageGalleryController: Image gallery navigation and selection
    AppController: Main application controller and workflow coordination
    MultiDockController: Multi-panel interface management
    EditImageController: HDR image editing controls and parameters
    ImageInfoController: Image metadata and information display
    AdvanceSliderController: Advanced slider control with auto-adjustment
    ToneCurveController: Tone curve editing and B-spline management
    LightnessMaskController: Lightness mask controls for tone range selection
    HDRviewerController: HDR image display and comparison
    LchColorSelectorController: LCH color space editing interface
    GeometryController: Geometric transformation controls
    ImageAestheticsController: Image aesthetics analysis and display

Key Features:
    - Complete MVC architecture for HDR image editing
    - Multi-threaded processing with progress indicators
    - HDR display management and calibration
    - Advanced color editing with LCH color space support
    - Tone curve editing with B-spline interpolation
    - Batch processing and export capabilities
    - Real-time preview and comparison tools
"""
# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------

import enum, sys, subprocess, copy, colour, time, os, shutil, datetime, ctypes
import numpy as np
# pyQT5 import
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtWidgets import QMessageBox

from . import model, view, thread
import hdrCore.image, hdrCore.processing, hdrCore.utils
import hdrCore.coreC
import preferences.preferences as pref

# zj add for semi-auto curve 
import torch
from hdrCore.net import Net
from torch.autograd import Variable

# -----------------------------------------------------------------------------
# --- package methods ---------------------------------------------------------
# -----------------------------------------------------------------------------
def getScreenSize(app):
    """
    Get screen resolution information for multi-monitor support.
    
    Queries the application for all available screens and returns their
    dimensions for HDR display configuration and window management.
    
    Args:
        app (QApplication): PyQt5 application instance
        
    Returns:
        list: List of QSize objects representing screen dimensions
    """
    screens = app.screens()
    res = list(map(lambda x: x.size(), screens))
    return res
# -----------------------------------------------------------------------------
# --- Class GalleryMode -------------------------------------------------------
# -----------------------------------------------------------------------------
class GalleryMode(enum.Enum):
    """
    Enumeration for gallery display layout configurations.
    
    Defines different grid layouts for the image gallery, allowing users
    to view multiple images simultaneously in various arrangements.
    Each mode specifies the number of rows and columns for image display.
    
    Attributes:
        _1x1 (int): Single image view (1 row, 1 column)
        _3x2 (int): 3×2 grid layout (2 rows, 3 columns)
        _6x4 (int): 6×4 grid layout (4 rows, 6 columns)
        _9x6 (int): 9×6 grid layout (6 rows, 9 columns)
        _2x1 (int): 2×1 layout (1 row, 2 columns) for side-by-side comparison
    
    Static Methods:
        nbRow: Get number of rows for a gallery mode
        nbCol: Get number of columns for a gallery mode
    """

    _1x1         = 0         # 
    _3x2         = 1         # 
    _6x4         = 2         # 
    _9x6         = 3         #
    
    _2x1        =  4

    def nbRow(m):
        """
        Get the number of rows for a gallery mode.
        
        Args:
            m (GalleryMode): Gallery mode enumeration value
            
        Returns:
            int: Number of rows in the gallery layout
        """
        if m == GalleryMode._1x1: return 1
        if m == GalleryMode._3x2: return 2
        if m == GalleryMode._6x4: return 4
        if m == GalleryMode._9x6: return 6

        if m == GalleryMode._2x1: return 1
    def nbCol(m):
        """
        Get the number of columns for a gallery mode.
        
        Args:
            m (GalleryMode): Gallery mode enumeration value
            
        Returns:
            int: Number of columns in the gallery layout
        """
        if m == GalleryMode._1x1: return 1
        if m == GalleryMode._3x2: return 3
        if m == GalleryMode._6x4: return 6
        if m == GalleryMode._9x6: return 9

        if m == GalleryMode._2x1: return 2
# -----------------------------------------------------------------------------
# --- Class ImageWidgetController ---------------------------------------------
# -----------------------------------------------------------------------------
class ImageWidgetController:
    """
    Controller for individual image widget components.
    
    Manages the display and interaction of individual image widgets within
    the gallery and editing interface. Handles image loading, display
    updates, and user interactions for single image components.
    
    Attributes:
        model (model.ImageWidgetModel): Data model for the image widget
        view (view.ImageWidgetView): View component for image display
        _id (int): Unique identifier for the widget
    
    Methods:
        setImage: Update the displayed image
        setQPixmap: Set the display pixmap directly
        id: Get the widget's unique identifier
    """

    def __init__(self, image=None,id = -1):
        """
        Initialize an image widget controller.
        
        Args:
            image (numpy.ndarray or hdrCore.image.Image, optional): Initial image to display
            id (int, optional): Unique identifier for the widget (default: -1)
        """

        self.model = model.ImageWidgetModel(self)
        self.view = view.ImageWidgetView(self)

        self._id = id # store an (unique) id 

        if isinstance(image, (np.ndarray, hdrCore.image.Image)):
            self.model.setImage(image)
            self.view.setPixmap(self.model.getColorData())

    def setImage(self, image):
        """
        Update the image displayed in the widget.
        
        Args:
            image (numpy.ndarray or hdrCore.image.Image): New image to display
            
        Returns:
            QPixmap: Updated pixmap for the display
        """
        self.model.setImage(image)
        return self.view.setPixmap(self.model.getColorData())

    def setQPixmap(self, qPixmap):
        """
        Set the display pixmap directly.
        
        Args:
            qPixmap (QPixmap): Pixmap to display in the widget
        """
        self.view.setQPixmap(qPixmap)

    def id(self): 
        """
        Get the widget's unique identifier.
        
        Returns:
            int: Unique identifier for this widget
        """
        return self._id
# -----------------------------------------------------------------------------
# --- Class ImageGalleryController --------------------------------------------
# -----------------------------------------------------------------------------
class ImageGalleryController():
    """
    Controller for the image gallery interface.
    
    Manages the gallery view that displays multiple images in grid layouts,
    handles pagination, gallery mode switching, and image selection. Provides
    the main interface for browsing and selecting images for editing.
    
    The gallery supports multiple display modes (1×1, 3×2, 6×4, 9×6, 2×1) and
    handles pagination when there are more images than can fit in the current view.
    
    Attributes:
        parent: Parent controller (typically AppController)
        view (view.ImageGalleryView): Gallery view component
        model (model.ImageGalleryModel): Gallery data model
    
    Methods:
        setImages: Load images into the gallery
        updateImages: Refresh the gallery display
        selectImage: Handle image selection
        getSelectedProcessPipe: Get the processing pipeline for selected image
        save: Save the current selection
        
    Gallery Mode Methods:
        callBackButton_1x1: Switch to single image view
        callBackButton_3x2: Switch to 3×2 grid view
        callBackButton_6x4: Switch to 6×4 grid view
        callBackButton_9x6: Switch to 9×6 grid view
        callBackButton_2x1: Switch to 2×1 comparison view
        
    Navigation Methods:
        callBackButton_previousPage: Navigate to previous page
        callBackButton_nextPage: Navigate to next page
        computePageNumberOnGalleryModeChange: Calculate page after mode change
    """

    def __init__(self, parent):
        """
        Initialize the image gallery controller.
        
        Args:
            parent: Parent controller that manages this gallery
        """
        if pref.verbose: print(" [CONTROL] >> ImageGalleryController.__init__()")

        self.parent = parent    # AppView
        self.view = view.ImageGalleryView(self)
        self.model = model.ImageGalleryModel(self)

    def setImages(self, imageFiles): 
        """
        Load a list of image files into the gallery.
        
        Args:
            imageFiles (list): List of image file paths to load
        """
        self.model.setImages(imageFiles)

    def updateImages(self):
        """
        Refresh the gallery display with current images.
        
        Called by the model when image data changes to update the view.
        Resets the page number and refreshes the display.
        """
        self.view.pageNumber = 0 # reset page number
        self.view.updateImages()

    def callBackButton_previousPage(self):  
        """
        Navigate to the previous page in the gallery.
        
        Handles pagination when there are more images than can fit in the
        current gallery view. In single image mode, also selects the image.
        """
        self.view.changePageNumber(-1)
        if self.view.shapeMode == GalleryMode._1x1 : self.selectImage(0)

    def callBackButton_nextPage(self):      
        """
        Navigate to the next page in the gallery.
        
        Handles pagination when there are more images than can fit in the
        current gallery view. In single image mode, also selects the image.
        """
        self.view.changePageNumber(+1)
        if self.view.shapeMode == GalleryMode._1x1 : self.selectImage(0)

    def computePageNumberOnGalleryModeChange(self,newGalleryMode):
        """
        Calculate the appropriate page number when changing gallery modes.
        
        When switching between different gallery layouts, this method ensures
        the currently selected or viewed image remains visible by calculating
        the correct page number for the new layout.
        
        Args:
            newGalleryMode (GalleryMode): Target gallery layout mode
            
        Returns:
            int: Calculated page number for the new gallery mode
        """
        currentPage = self.view.pageNumber
        nbImagePerPage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        selectedImage = self.model.selectedImage() if (self.model.selectedImage()!=-1) else currentPage*nbImagePerPage
        newNbImagePerPage = GalleryMode.nbRow(newGalleryMode)*GalleryMode.nbCol(newGalleryMode)

        newPageNumber = selectedImage//newNbImagePerPage

        return newPageNumber

    def callBackButton_1x1(self):
        """
        Switch gallery to single image view mode (1×1).
        
        Changes the gallery layout to display one image at a time, suitable
        for detailed viewing and editing. Recalculates pagination and updates
        the display.
        """
        if self.view.shapeMode != GalleryMode._1x1:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._1x1)
            self.view.shapeMode = GalleryMode._1x1
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_3x2(self): 
        """
        Switch gallery to 3×2 grid view mode.
        
        Changes the gallery layout to display 6 images in a 3×2 grid,
        suitable for browsing multiple images while maintaining reasonable
        image size for preview.
        """
        if self.view.shapeMode != GalleryMode._3x2:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._3x2)
            self.view.shapeMode = GalleryMode._3x2
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_6x4(self): 
        """
        Switch gallery to 6×4 grid view mode.
        
        Changes the gallery layout to display 24 images in a 6×4 grid,
        suitable for quick browsing of large image collections with
        smaller thumbnail sizes.
        """
        if self.view.shapeMode != GalleryMode._6x4:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._6x4)
            self.view.shapeMode = GalleryMode._6x4
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_9x6(self):
        """
        Switch gallery to 9×6 grid view mode.
        
        Changes the gallery layout to display 54 images in a 9×6 grid,
        suitable for overview browsing of very large image collections
        with small thumbnail sizes.
        """
        if self.view.shapeMode != GalleryMode._9x6:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._9x6)
            self.view.shapeMode = GalleryMode._9x6
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_2x1(self):
        """
        Switch gallery to 2×1 comparison view mode.
        
        Changes the gallery layout to display 2 images side by side,
        ideal for comparing different processing results or comparing
        before/after versions of images.
        """
        if self.view.shapeMode != GalleryMode._2x1:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._2x1)
            self.view.shapeMode = GalleryMode._2x1
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def selectImage(self, id):
        """
        Handle selection of an image in the gallery.
        
        Processes user selection of an image and updates the editing interface
        to work with the selected image's processing pipeline.
        
        Args:
            id (int): Index of the selected image within the current page
        """
        if pref.verbose: print(" [CONTROL] >> ImageGalleryController.selectImage()")

        nbImagePage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        idxImage = self.view.pageNumber*nbImagePage+id
        # check id
        if (idxImage < len(self.model.processPipes)):
            # update selected image
            processPipe = self.model.processPipes[idxImage]
            if processPipe:
                if self.parent.dock.setProcessPipe(processPipe):
                    self.model.setSelectedImage(idxImage)

    def getSelectedProcessPipe(self):
        if pref.verbose:  print(" [CONTROL] >> ImageGalleryController.getSelectedProcessPipe()")
        return self.model.getSelectedProcessPipe()

    def setProcessPipeWidgetQPixmap(self, qPixmap):
        idxProcessPipe = self.model.selectedImage()
        nbImagePage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        idxImageWidget = idxProcessPipe%nbImagePage
        if pref.verbose:  print(" >> ImageGalleryController.setProcessPipeWidgetQPixmap(...)[ image id:",idxProcessPipe,">> image widget controller:",idxImageWidget,"]")
        self.view.imagesControllers[idxImageWidget].setQPixmap(qPixmap)

    def save(self):
        if pref.verbose: print(" [CONTROL] >> ImageGalleryController.save()")
        self.model.save()

    def currentPage(self): return self.view.currentPage()

    def pageIdx(self):
        nb = self.currentPage()
        nbImagePage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        return (nb*nbImagePage), ((nb+1)*nbImagePage)

    def getFilenamesOfCurrentPage(self): return self.model.getFilenamesOfCurrentPage()

    def getProcessPipeById(self,i) : return self.model.getProcessPipeById(i)

    def getProcessPipes(self): return self.model.processPipes
# -----------------------------------------------------------------------------
# --- Class AppController -----------------------------------------------------
# -----------------------------------------------------------------------------
class AppController(object):
    """
    Main application controller implementing the MVC pattern.
    
    This class serves as the central controller for the uHDR application,
    managing the main window, coordinating between different components,
    and handling high-level application operations like file management,
    HDR display, and image export functionality.
    
    The AppController orchestrates interactions between the image gallery,
    editing interface, HDR viewer, and manages the application's workflow
    from image loading to final export operations.
    
    Attributes:
        screenSize (list): Available screen dimensions for multi-monitor support
        hdrDisplay (HDRviewerController): HDR image display controller
        view (view.AppView): Main application view component
        model (model.AppModel): Main application data model
        dirName (str): Currently selected directory path
        imagesName (list): List of image filenames in current directory
    
    Methods:
        callBackSelectDir: Open directory selection dialog and load images
        callBackSave: Save current image metadata and processing state
        callBackQuit: Clean application shutdown with data preservation
        callBackDisplayHDR: Display processed HDR image on HDR monitor
        callBackEndDisplay: Complete HDR display processing workflow
        callBackCloseDisplayHDR: Close HDR display and return to splash
        callBackCompareRawEditedHDR: Side-by-side comparison of original vs edited
        callBackExportHDR: Export single HDR image with processing applied
        callBackEndExportHDR: Complete single HDR export workflow
        callBackExportAllHDR: Batch export all processed HDR images
        callBackEndAllExportHDR: Complete batch HDR export workflow
    """

    def __init__(self, app):
        """
        Initialize the main application controller.
        
        Sets up the application's core components including screen detection,
        HDR display initialization, and main view creation. Establishes the
        foundation for the complete HDR editing workflow.
        
        Args:
            app (QApplication): PyQt5 application instance for screen detection
        """
        if pref.verbose: print(" [CONTROL] >> AppController.__init__()")

        self.screenSize = getScreenSize(app)# get screens size

        # attributes
        self.hdrDisplay = HDRviewerController(self)
        self.view =  view.AppView(self, HDRcontroller = self.hdrDisplay)                         
        self.model = model.AppModel(self)

        self.dirName = None
        self.imagesName = []
        
        self.view.show()
    # -----------------------------------------------------------------------------

    def callBackSelectDir(self):
        """
        Open directory selection dialog and load HDR images.
        
        Presents a file dialog for directory selection, discovers all supported
        image files in the chosen directory, and loads them into the gallery.
        Automatically saves any pending metadata before switching directories.
        
        The method handles the complete workflow of:
        1. Directory selection via QFileDialog
        2. Saving current image metadata
        3. Discovering supported image formats
        4. Loading images into the gallery view
        5. Resetting HDR display to splash screen
        """
        if pref.verbose: print(" [CONTROL] >> AppController.callBackSelectDir()")
        dirName = QFileDialog.getExistingDirectory(None, 'Select Directory', self.model.directory)
        if dirName != "":
            # save current images (metadata)
            self.view.imageGalleryController.save()
            # get images in the selected directory
            self.imagesName = []; self.imagesName = list(self.model.setDirectory(dirName))

            self.view.imageGalleryController.setImages(self.imagesName)
            self.hdrDisplay.displaySplash()
    # -----------------------------------------------------------------------------
    def callBackSave(self): 
        """
        Save current processing state and metadata.
        
        Triggers the gallery controller to save all current image metadata
        and processing pipeline states to persistent storage.
        """
        self.view.imageGalleryController.save()
    # -----------------------------------------------------------------------------
    def callBackQuit(self):
        """
        Handle application shutdown with proper cleanup.
        
        Performs clean application exit by saving all metadata, closing
        HDR display processes, and terminating the application gracefully.
        Ensures no data loss during application closure.
        """
        if pref.verbose: print(" [CB] >> AppController.callBackQuit()")
        self.view.imageGalleryController.save()
        self.hdrDisplay.close()
        sys.exit()
    # -----------------------------------------------------------------------------
    def callBackDisplayHDR(self):
        """
        Display processed HDR image on external HDR monitor.
        
        Initiates the HDR display workflow by processing the currently selected
        image at full resolution with the complete processing pipeline, then
        sends the result to the external HDR display. Shows progress feedback
        during the compute-intensive full-resolution processing.
        
        The workflow includes:
        1. Retrieving selected processing pipeline
        2. Saving processing metadata
        3. Loading full-resolution source image
        4. Resizing to display resolution
        5. Applying complete processing pipeline
        6. Threading the computation with progress updates
        """

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackDisplayHDR()")

        selectedProcessPipe = self.view.imageGalleryController.model.getSelectedProcessPipe()

        if selectedProcessPipe:
            self.view.statusBar().showMessage('displaying HDR image, full size image computation: start, please wait !')
            self.view.statusBar().repaint()
            # save current processpipe metada
            originalImage = copy.deepcopy(selectedProcessPipe.originalImage)
            originalImage.metadata.metadata['processpipe'] = selectedProcessPipe.toDict()
            originalImage.metadata.save()

            # load full size image
            img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = False 
            # make a copy of selectedProcessPipe  
            processpipe = copy.deepcopy(selectedProcessPipe)

            # set size to display size
            size = pref.getDisplayShape()
            img = img.process(hdrCore.processing.resize(),size=(None, size[1]))

            # set image to process-pipe
            processpipe.setImage(img)

            thread.cCompute(self.callBackEndDisplay, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)
    # -----------------------------------------------------------------------------
    def callBackEndDisplay(self, img):
        """
        Complete HDR display processing and send to external monitor.
        
        Finalizes the HDR display workflow by applying final processing steps
        and sending the image to the external HDR display. Updates the status
        bar to indicate completion.
        
        Args:
            img (hdrCore.image.Image): Processed HDR image ready for display
            
        Note:
            Re-enables auto-resizing after full-resolution processing and
            applies display-specific scaling before HDR monitor output.
        """

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackEndDisplay()")

        # turn off: autoResize
        hdrCore.processing.ProcessPipe.autoResize = True  
        
        self.view.statusBar().showMessage('displaying HDR image, full size image computation: done !')

        # clip, scale
        img = img.process(hdrCore.processing.clip())
        img.colorData = img.colorData*pref.getDisplayScaling()

        colour.write_image(img.colorData,"temp.hdr", method='Imageio') # local copy for display
        self.hdrDisplay.displayFile("temp.hdr")
    # -----------------------------------------------------------------------------
    def callBackCloseDisplayHDR(self):
        """
        Close HDR display and return to splash screen.
        
        Terminates the current HDR display session and returns the external
        HDR monitor to the default splash screen state.
        """
        if pref.verbose: print(" [CONTROL] >> AppController.callBackCloseDisplayHDR()")
        self.hdrDisplay.displaySplash()
    # -----------------------------------------------------------------------------
    def callBackCompareRawEditedHDR(self):
        """
        Display side-by-side comparison of original and edited HDR images.
        
        Creates a comparison view showing the original unprocessed image
        alongside the current edited version. Both images are displayed
        at the same scale on the HDR monitor for direct visual comparison.
        
        The comparison workflow:
        1. Loads original unprocessed image
        2. Applies current processing pipeline
        3. Resizes both images to fit side-by-side
        4. Creates composite comparison image
        5. Displays on HDR monitor with proper scaling
        """

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackCompareOriginalInputHDR()")

        # process real size image
        # get selected process pipe
        selectedProcessPipe = self.view.imageGalleryController.model.getSelectedProcessPipe()

        if selectedProcessPipe:         # check if a process pipe is selected

            # read original image
            img = hdrCore.image.Image.read(selectedProcessPipe.originalImage.path+'/'+selectedProcessPipe.originalImage.name)

            # resize
            screenY, screenX = pref.getDisplayShape()

            imgY, imgX,_ = img.shape

            marginY = int((screenY - imgY/2)/2)
            marginX = int(marginY/4)
            imgXp = int((screenX - 3*marginX)/2)
            img = img.process(hdrCore.processing.resize(),size=(None,imgXp))
            imgY, imgX, _ = img.shape

            # original image after resize
            ori = copy.deepcopy(img)

            # build process pipe from selected one them compute
            pp = hdrCore.processing.ProcessPipe()
            hdrCore.processing.ProcessPipe.autoResize = False   # stop autoResize
            params= []
            for p in selectedProcessPipe.processNodes: 
                pp.append(copy.deepcopy(p.process),paramDict=None, name=copy.deepcopy(p.name))
                params.append({p.name:p.params})
            img.metadata.metadata['processpipe'] = params
            pp.setImage(img)

            res = hdrCore.coreC.coreCcompute(img, pp)
            res = res.process(hdrCore.processing.clip())
            
            imgYres, imgXres, _ = res.colorData.shape

            hdrCore.processing.ProcessPipe.autoResize = True    # return to autoResize

            # make comparison image
            oriColorData = ori.colorData*pref.getDisplayScaling()
            resColorData = res.colorData*pref.getDisplayScaling()
            display = np.ones((screenY,screenX,3))*0.2
            marginY = int((screenY - imgY)/2)
            marginYres = int((screenY - imgYres)/2)

            display[marginY:marginY+imgY, marginX:marginX+imgX,:] = oriColorData
            display[marginYres:marginYres+imgYres, 2*marginX+imgX:2*marginX+imgX+imgXres,:] = resColorData
            
            # save as compOrigFinal.hdr
            colour.write_image(display,'compOrigFinal.hdr', method='Imageio')
            self.hdrDisplay.displayFile('compOrigFinal.hdr')
    # -----------------------------------------------------------------------------
    def callBackExportHDR(self):
        """
        Export single HDR image with current processing applied.
        
        Initiates the HDR export workflow for the currently selected image,
        allowing the user to choose the export destination and applying
        the complete processing pipeline at full resolution.
        
        The export process includes:
        1. User directory selection for output
        2. Full-resolution image loading
        3. Complete processing pipeline application
        4. HDR-specific encoding and scaling
        5. File output with appropriate metadata
        """

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackExportHDR()") 

        selectedProcessPipe = self.view.imageGalleryController.model.getSelectedProcessPipe()


        if selectedProcessPipe:
            # select dir where to save export
            self.dirName = QFileDialog.getExistingDirectory(None, 'Select Directory where to export HDR file', self.model.directory)

            # show export message
            self.view.statusBar().showMessage('exporting HDR image ('+pref.getHDRdisplay()['tag']+'), full size image computation: start, please wait !')
            self.view.statusBar().repaint()

            # save current processpipe metada
            originalImage = copy.deepcopy(selectedProcessPipe.originalImage)
            originalImage.metadata.metadata['processpipe'] = selectedProcessPipe.toDict()
            originalImage.metadata.save()

            # load full size image
            img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = False 
            # make a copy of selectedProcessPipe  
            processpipe = copy.deepcopy(selectedProcessPipe)

            # set image to process-pipe
            processpipe.setImage(img)

            thread.cCompute(self.callBackEndExportHDR, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)
    # -----------------------------------------------------------------------------
    def callBackEndExportHDR(self, img):
        """
        Complete single HDR export and save to disk.
        
        Finalizes the HDR export process by applying final processing,
        saving to the selected directory, and optionally displaying
        the result on the HDR monitor.
        
        Args:
            img (hdrCore.image.Image): Processed HDR image ready for export
        """
        # turn off: autoResize
        hdrCore.processing.ProcessPipe.autoResize = True  
        
        self.view.statusBar().showMessage('exporting HDR image ('+pref.getHDRdisplay()['tag']+'), full size image computation: done !')

        # clip, scale
        img = img.process(hdrCore.processing.clip())
        img.colorData = img.colorData*pref.getDisplayScaling()

        if self.dirName:
            pathExport = os.path.join(self.dirName, img.name[:-4]+pref.getHDRdisplay()['post']+'.hdr')
            img.type = hdrCore.image.imageType.HDR
            img.metadata.metadata['processpipe'] = None
            img.metadata.metadata['display'] = pref.getHDRdisplay()['tag']

            img.write(pathExport)

        colour.write_image(img.colorData,"temp.hdr", method='Imageio') # local copy for display
        self.hdrDisplay.displayFile("temp.hdr")
    # -----------------------------------------------------------------------------
    def callBackExportAllHDR(self):
        """
        Initiate batch export of all HDR images in the gallery.
        
        Starts the batch export process for all images in the current gallery,
        allowing users to export multiple processed HDR images with a single
        operation. Provides progress feedback during the batch operation.
        """
        if pref.verbose:  print(" [CONTROL] >> AppController.callBackExportAllHDR()")

        self.processPipes = self.view.imageGalleryController.getProcessPipes()

        # select dir where to save export
        self.dirName = QFileDialog.getExistingDirectory(None, 'Select Directory where to export HDR file', self.model.directory)
        self.view.statusBar().showMessage('exporting '+str(len(self.processPipes))+' HDR images ... please wait')
        self.view.statusBar().repaint()
        self.imageToExport = len(self.processPipes) ; self.imageExportDone = 0

        pp = self.processPipes[0]

        # save current processpipe metada
        originalImage = copy.deepcopy(pp.originalImage)
        originalImage.metadata.metadata['processpipe'] = pp.toDict()
        originalImage.metadata.save()

        # load full size image
        img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

        # turn off: autoResize
        hdrCore.processing.ProcessPipe.autoResize = False 
        # make a copy of selectedProcessPipe  
        processpipe = copy.deepcopy(pp)

        # set image to process-pipe
        processpipe.setImage(img)

        thread.cCompute(self.callBackEndAllExportHDR, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)            
    # -----------------------------------------------------------------------------
    def callBackEndAllExportHDR(self, img):
        """
        Process single image in batch export and continue to next.
        
        Handles the completion of each individual image in the batch export
        process, saves the current image, updates progress, and initiates
        processing of the next image in the queue.
        
        Args:
            img (hdrCore.image.Image): Current processed HDR image in batch
            
        Note:
            Recursively processes all images in the batch until completion,
            providing progress updates throughout the operation.
        """
        # last image ?
        self.imageExportDone +=1

        self.view.statusBar().showMessage('exporting HDR images ('+pref.getHDRdisplay()['tag']+'):'+str(int(100*self.imageExportDone/self.imageToExport))+'% done !')

        # clip, scale
        img = img.process(hdrCore.processing.clip())
        img.colorData = img.colorData*pref.getDisplayScaling()

        if self.dirName:
            pathExport = os.path.join(self.dirName, img.name[:-4]+pref.getHDRdisplay()['post']+'.hdr')
            img.type = hdrCore.image.imageType.HDR
            img.metadata.metadata['processpipe'] = None
            img.metadata.metadata['display'] = pref.getHDRdisplay()['tag']

            img.write(pathExport)

        if self.imageExportDone == self.imageToExport :
            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = True
        else:
            pp = self.processPipes[self.imageExportDone]

            if not pp:
                img = hdrCore.image.Image.read(self.imagesName[self.imageExportDone], thumb=True)
                pp = model.EditImageModel.buildProcessPipe()
                pp.setImage(img)                      


            # save current processpipe metada
            originalImage = copy.deepcopy(pp.originalImage)
            originalImage.metadata.metadata['processpipe'] = pp.toDict()
            originalImage.metadata.save()

            # load full size image
            img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = False 
            # make a copy of selectedProcessPipe  
            processpipe = copy.deepcopy(pp)

            # set image to process-pipe
            processpipe.setImage(img)

            thread.cCompute(self.callBackEndAllExportHDR, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)            
# ------------------------------------------------------------------------------------------
# --- class MultiDockController() ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class MultiDockController():
    """
    Controller for managing multiple docked panels in the interface.
    
    This class manages the docking system that allows switching between different
    functional panels: EDIT (image editing), INFO (metadata/information), and
    MIAM (aesthetics analysis). It provides seamless transitions between
    different modes of interaction with the HDR images.
    
    The docking system allows users to focus on specific aspects of image
    processing while maintaining a consistent workflow and shared processing
    pipeline state across all panels.
    
    Attributes:
        parent: Parent controller (typically AppController)
        view (view.MultiDockView): Multi-panel view component
        model: Data model (currently None, uses child controllers' models)
    
    Methods:
        activateEDIT: Switch to image editing panel
        activateINFO: Switch to image information/metadata panel  
        activateMIAM: Switch to image aesthetics analysis panel
        switch: Generic panel switching mechanism
        setProcessPipe: Propagate processing pipeline to active panel
    """
    def __init__(self,parent=None, HDRcontroller = None):
        """
        Initialize the multi-dock controller.
        
        Args:
            parent: Parent controller for coordination
            HDRcontroller: HDR display controller for HDR preview integration
        """
        if pref.verbose: print(" [CONTROL] >> MultiDockController.__init__()")

        self.parent = parent
        self.view = view.MultiDockView(self, HDRcontroller)
        self.model = None
    # ---------------------------------------------------------------------------------------
    def activateEDIT(self): 
        """
        Activate the image editing panel.
        
        Switches the interface to the image editing mode, providing access
        to exposure, contrast, tone curves, color editing, and other HDR
        processing controls.
        """
        self.switch(0)
    def activateINFO(self): 
        """
        Activate the image information panel.
        
        Switches the interface to the metadata and information display mode,
        showing EXIF data, processing history, and image properties.
        """
        self.switch(1)
    def activateMIAM(self):  
        """
        Activate the aesthetics analysis panel.
        
        Switches the interface to the MIAM (aesthetics) analysis mode,
        displaying image quality metrics, color palette analysis, and
        aesthetic scoring information.
        """
        self.switch(2)
    # ---------------------------------------------------------------------------------------
    def switch(self,nb):
        """
        Switch to the specified panel number.
        
        Args:
            nb (int): Panel index (0=EDIT, 1=INFO, 2=MIAM)
        """
        if pref.verbose:  print(" [CONTROL] >> MultiDockController.switch()")
        self.view.switch(nb)
    # --------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        """
        Set the processing pipeline for the active panel.
        
        Propagates the processing pipeline to the currently active panel,
        ensuring that all panels work with the same image and processing state.
        
        Args:
            processPipe: Processing pipeline to apply to the active panel
            
        Returns:
            bool: True if the pipeline was successfully set, False otherwise
        """
        if pref.verbose: print(" [CONTROL] >> MultiDockController.setProcessPipe(",processPipe.getImage().name,")")

        return self.view.setProcessPipe(processPipe)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class EditImageController:
    """
    Controller for HDR image editing interface and processing pipeline.
    
    This class manages the comprehensive HDR image editing workflow, providing
    controls for exposure, contrast, tone curves, saturation, color editing,
    geometry transformations, and processing pipeline management. It coordinates
    between the editing interface and the underlying HDR processing system.
    
    The controller handles real-time preview updates, auto-adjustment algorithms,
    parameter synchronization, and HDR display integration for immediate visual
    feedback during the editing process.
    
    Attributes:
        parent: Parent controller for coordination
        previewHDR (bool): Enable/disable HDR preview updates
        controllerHDR: HDR display controller for real-time preview
        view (view.EditImageView): Image editing interface view
        model (model.EditImageModel): Image editing data model and processing
    
    Methods:
        setProcessPipe: Set the processing pipeline for editing
        getProcessPipe: Get current processing pipeline
        buildView: Rebuild the editing interface
        autoExposure: Calculate and apply optimal exposure
        changeExposure: Adjust image exposure in EV stops
        changeContrast: Adjust image contrast
        changeToneCurve: Modify tone curve with control points
        changeLightnessMask: Configure lightness-based masking
        changeSaturation: Adjust color saturation
        changeColorEditor: Apply selective color editing
        changeGeometry: Apply geometric transformations
        updateImage: Handle processing completion and view updates
    """

    def __init__(self, parent=None, HDRcontroller = None):
        """
        Initialize the image editing controller.
        
        Args:
            parent: Parent controller for coordination
            HDRcontroller: HDR display controller for real-time preview
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.__init__(",")")

        self.parent = parent

        self.previewHDR = True
        self.controllerHDR = HDRcontroller

        self.view = view.EditImageView(self)
        self.model = model.EditImageModel(self)
    # -----------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        """
        Set the processing pipeline for editing.
        
        Configures the editing interface to work with the specified processing
        pipeline, updating all controls and displays to reflect the current
        processing state.
        
        Args:
            processPipe: Processing pipeline containing image and processing nodes
            
        Returns:
            bool: True if the pipeline was successfully set, False otherwise
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.setProcessPipe(",")")

        if self.model.setProcessPipe(processPipe):

            # update view
            self.view.setProcessPipe(processPipe)
            self.view.imageWidgetController.setImage(processPipe.getImage())

            self.view.plotToneCurve()

            # update hdr viewer
            self.controllerHDR.displaySplash()

            return True
        else:
            return False
    # -----------------------------------------------------------------------------
    def getProcessPipe(self) : 
        """
        Get the current processing pipeline.
        
        Returns:
            ProcessPipe: Current processing pipeline with all applied operations
        """
        return self.model.getProcessPipe()
    # -----------------------------------------------------------------------------
    def buildView(self,processPipe=None):
        """
        Build or rebuild the editing interface view.
        
        Reconstructs the editing interface, typically called when switching
        between different panels or when the interface needs to be refreshed.
        
        Args:
            processPipe: Optional processing pipeline to set during build
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.buildView(",")")

        """ called when MultiDockController recall a controller/view """
        self.view = view.EditImageView(self, build=True)
        if processPipe: self.setProcessPipe(processPipe)
    # -----------------------------------------------------------------------------
    def autoExposure(self): 
        """
        Calculate and apply optimal exposure automatically.
        
        Uses histogram analysis to determine the optimal exposure value that
        maximizes dynamic range usage without clipping, then applies this
        exposure and updates the interface controls.
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.autoExposure(",")")
        if self.model.processpipe:
      
            img = self.model.autoExposure()

            # update view with computed EV
            paramDict = self.model.getEV()
            self.view.exposure.setValue(paramDict['EV'])

            qPixmap =  self.view.setImage(img)
            self.parent.controller.parent.controller.view.imageGalleryController.setProcessPipeWidgetQPixmap(qPixmap)
    # -----------------------------------------------------------------------------
    def changeExposure(self,value):
        """
        Adjust image exposure by specified EV value.
        
        Args:
            value (float): Exposure adjustment in EV stops (positive = brighter)
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeExposure(",value,")")
        if self.model.processpipe: self.model.changeExposure(value)
    # -----------------------------------------------------------------------------
    def changeContrast(self,value):
        """
        Adjust image contrast by specified amount.
        
        Args:
            value (float): Contrast adjustment value (positive = more contrast)
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeContrast(",value,")")
        if self.model.processpipe: self.model.changeContrast(value)
    # -----------------------------------------------------------------------------
    def changeToneCurve(self,controlPoints):
        """
        Modify tone curve using B-spline control points.
        
        Args:
            controlPoints (dict): Control points defining the tone curve shape
                                 with keys like 'shadows', 'highlights', etc.
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeToneCurve("")")
        if self.model.processpipe: self.model.changeToneCurve(controlPoints)
    # -----------------------------------------------------------------------------
    def changeLightnessMask(self, maskValues):
        """
        Configure lightness-based masking for tone range selection.
        
        Args:
            maskValues (dict): Mask configuration for different tone ranges
                              (shadows, blacks, mediums, whites, highlights)
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeLightnessMask(",maskValues,")")
        if self.model.processpipe: self.model.changeLightnessMask(maskValues)
    # -----------------------------------------------------------------------------
    def changeSaturation(self,value):
        """
        Adjust color saturation globally.
        
        Args:
            value (float): Saturation adjustment value (positive = more saturated)
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeSaturation(",value,")")
        if self.model.processpipe: self.model.changeSaturation(value)
    # -----------------------------------------------------------------------------
    def changeColorEditor(self,values, idName):
        """
        Apply selective color editing to specific color ranges.
        
        Args:
            values (dict): Color editing parameters including selection criteria
                          and editing adjustments (hue, saturation, exposure, etc.)
            idName (str): Identifier for the color editor instance
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeColorEditor(",values,")")
        if self.model.processpipe: self.model.changeColorEditor(values, idName)
    # -----------------------------------------------------------------------------
    def changeGeometry(self,values):
        """
        Apply geometric transformations (crop, rotation, etc.).
        
        Args:
            values (dict): Geometry parameters including crop ratios,
                          rotation angles, and positioning adjustments
        """
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeGeometry(",values,")")
        if self.model.processpipe: self.model.changeGeometry(values)
    # -----------------------------------------------------------------------------
    def updateImage(self,imgTM):
        """
        Handle processing completion and update displays.
        
        Called when the processing pipeline computation is complete, updates
        the interface display, gallery thumbnail, and optionally triggers
        HDR preview updates.
        
        Args:
            imgTM (hdrCore.image.Image): Tone-mapped image for GUI display
        """
        qPixmap =  self.view.setImage(imgTM)
        self.parent.controller.parent.controller.view.imageGalleryController.setProcessPipeWidgetQPixmap(qPixmap)
        self.view.plotToneCurve()

        # if aesthetics model > notify required update


        if self.previewHDR and self.model.autoPreviewHDR:
            self.controllerHDR.callBackUpdate()
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageInfoController:
    """
    Controller for image information and metadata display interface.
    
    This class manages the display and editing of image metadata, EXIF information,
    processing history, and other image properties. It provides a comprehensive
    view of all technical information associated with HDR images.
    
    The controller handles metadata organization, user-controlled metadata
    visibility, and metadata modification workflows while maintaining
    data integrity throughout the editing process.
    
    Attributes:
        parent: Parent controller for coordination
        view (view.ImageInfoView): Metadata display interface
        model (model.ImageInfoModel): Metadata management model
        callBackActive (bool): Enable/disable callback processing
    
    Methods:
        setProcessPipe: Set processing pipeline for metadata display
        buildView: Rebuild the information interface
        metadataChange: Handle metadata visibility/editing changes
    """

    def __init__(self, parent=None):
        """
        Initialize the image information controller.
        
        Args:
            parent: Parent controller for coordination
        """
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.__init__()")

        self.parent = parent
        self.view = view.ImageInfoView(self)
        self.model = model.ImageInfoModel(self)

        self.callBackActive = True
    # -----------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        """
        Set processing pipeline for metadata display.
        
        Configures the information interface to display metadata and properties
        for the specified processing pipeline's image.
        
        Args:
            processPipe: Processing pipeline containing image with metadata
            
        Returns:
            bool: True (always successful for info display)
        """
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.setProcessPipe(",processPipe.getImage().name,")")
        self.model.setProcessPipe(processPipe)
        self.view.setProcessPipe(processPipe)
        return True
    # -----------------------------------------------------------------------------
    def buildView(self,processPipe=None):
        """
        Build or rebuild the information interface view.
        
        Reconstructs the metadata display interface, typically called when
        switching panels or refreshing the information display.
        
        Args:
            processPipe: Optional processing pipeline to set during build
        """
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.buildView()")

        """ called when MultiDockController recall a controller/view """
        self.view = view.ImageInfoView(self)
        if processPipe: self.setProcessPipe(processPipe)
    # -----------------------------------------------------------------------------
    def metadataChange(self,metaGroup,metaTag, on_off): 
        """
        Handle changes to metadata visibility or editing state.
        
        Processes user interactions with metadata display controls, allowing
        users to show/hide specific metadata groups or modify metadata values.
        
        Args:
            metaGroup (str): Metadata group identifier (e.g., 'EXIF', 'processing')
            metaTag (str): Specific metadata tag within the group
            on_off (bool): Enable/disable state for the metadata element
        """
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.useCaseChange(",metaGroup,",", metaTag,",", on_off,")")
        self.model.changeMeta(metaGroup,metaTag, on_off)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceSliderController():
    """
    Controller for advanced slider controls with auto-adjustment capabilities.
    
    This class provides enhanced slider functionality with features like automatic
    value calculation, reset capabilities, and custom step sizes. It's used throughout
    the HDR editing interface for precise parameter control with additional
    automation features.
    
    The advanced slider combines manual adjustment with intelligent auto-adjustment
    algorithms, allowing users to fine-tune parameters manually or rely on
    automatic optimization algorithms.
    
    Attributes:
        parent: Parent controller for coordination
        view (view.AdvanceSliderView): Slider interface component
        model (model.AdvanceSliderModel): Slider data model
        step (float): Step size for slider increments
        defaultValue (float): Default/reset value for the slider
        range (tuple): Minimum and maximum values for the slider
        callBackActive (bool): Enable/disable callback processing
        callBackValueChange: Callback function for value changes
        callBackAutoPush: Callback function for auto-adjustment
    
    Methods:
        sliderChange: Handle manual slider value changes
        setValue: Set slider value programmatically
        reset: Reset slider to default value
        auto: Trigger automatic value calculation
    """
    def __init__(self, parent,name, defaultValue, range, step,callBackValueChange=None,callBackAutoPush= None):
        """
        Initialize the advanced slider controller.
        
        Args:
            parent: Parent controller for coordination
            name (str): Display name for the slider
            defaultValue (float): Default value for reset functionality
            range (tuple): (min, max) range for slider values
            step (float): Step size for value increments
            callBackValueChange: Callback function for value changes
            callBackAutoPush: Callback function for auto-adjustment
        """
        if pref.verbose: print(" [CONTROL] >> AdvanceSliderController.__init__(",") ")
        self.parent = parent

        self.view = view.AdvanceSliderView(self,  name, defaultValue, range, step)
        self.model = model.AdvanceSliderModel(self, value=defaultValue)

        self.step = step
        self.defaultValue = defaultValue
        self.range = range

        self.callBackActive = True
        self.callBackValueChange = callBackValueChange
        self.callBackAutoPush = callBackAutoPush
    # -----------------------------------------------------------------------------
    def sliderChange(self):
        """
        Handle manual slider value changes.
        
        Processes slider movement events, updates the model and display,
        and triggers the value change callback if active. Converts slider
        position to actual value using the configured step size.
        """

        value = self.view.slider.value()*self.step

        if pref.verbose: print(" [CB] >> AdvanceSliderController.sliderChange(",value,")[callBackActive:",self.callBackActive,"] ")

        self.model.value = value
        self.view.editValue.setText(str(value))
        if self.callBackActive and self.callBackValueChange: self.callBackValueChange(value)
    # -----------------------------------------------------------------------------
    def setValue(self, value, callBackActive = True):
        """
        Set slider value programmatically.
        
        Updates the slider position and display to reflect the specified value,
        with optional callback suppression for initialization or batch updates.
        
        Args:
            value (float): New value to set
            callBackActive (bool, optional): Whether to trigger callbacks (default: True)
        """
        if pref.verbose: print(" [CONTROL] >> AdvanceSliderController.setValue(",value,") ")

        """ set value value in 'model' range"""
        self.callBackActive = callBackActive
        self.view.slider.setValue(int(value/self.step))
        self.view.editValue.setText(str(value))
        self.model.setValue(int(value))

        self.callBackActive = True
    # -----------------------------------------------------------------------------
    def reset(self):
        """
        Reset slider to its default value.
        
        Restores the slider to its initial default value and triggers
        the value change callback to update dependent processing.
        """
        if pref.verbose : print(" [CB] >> AdvanceSliderController.reset(",") ")

        self.setValue(self.defaultValue,callBackActive = False)
        if self.callBackValueChange: self.callBackValueChange(self.defaultValue)
    # -----------------------------------------------------------------------------
    def auto(self):
        """
        Trigger automatic value calculation.
        
        Initiates the auto-adjustment algorithm by calling the registered
        auto-push callback, which typically analyzes the image and sets
        an optimal value automatically.
        """
        if pref.verbose: print(" [CB] >> AdvanceSliderController.auto(",") ")

        if self.callBackAutoPush: self.callBackAutoPush()
# ------------------------------------------------------------------------------------------
# --- class AdvanceSliderController --------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ToneCurveController():
    """
    Controller for tone curve editing with B-spline interpolation.
    
    This class manages the tone curve editing interface, allowing users to adjust
    the luminance mapping through control points that define a smooth B-spline curve.
    It includes automatic curve generation using machine learning, histogram overlay
    display, and comprehensive curve visualization options.
    
    The tone curve is a fundamental HDR processing tool that maps input luminance
    values to output values, effectively controlling the overall tone mapping
    and local contrast of the image.
    
    Attributes:
        parent: Parent controller for coordination
        model (model.ToneCurveModel): Tone curve mathematical model
        view (view.ToneCurveView): Curve editing interface
        callBackActive (bool): Enable/disable callback processing
        showInput (bool): Display input image histogram
        showbefore (bool): Display pre-curve histogram
        showAfter (bool): Display post-curve histogram  
        showOutput (bool): Display final output histogram
        weightFile (str): Machine learning model file for auto-curve
        networkModel: PyTorch neural network for automatic curve generation
    
    Methods:
        sliderChange: Handle control point slider adjustments
        setValues: Set all curve control points programmatically
        autoCurve: Generate curve automatically using ML analysis
        reset: Reset specific control point to default
        plotCurve: Display curve and histogram overlays
    """
    def __init__(self, parent):
        """
        Initialize the tone curve controller.
        
        Args:
            parent: Parent controller for coordination and image access
        """

        self.parent = parent
        self.model = model.ToneCurveModel()
        self.view = view.ToneCurveView(self)
        self.callBackActive = True

        # tone curve display control
        self.showInput =False
        self.showbefore = False
        self.showAfter = False
        self.showOutput = True

        # zj add semi-auto curve
        # machine learning network and weight file 
        self.weightFile = 'MSESig505_0419.pth'
        self.networkModel = None  
    # -----------------------------------------------------------------------------
    def sliderChange(self, key, value):
        """
        Handle tone curve control point adjustments.
        
        Processes changes to individual control points (shadows, blacks, mediums,
        whites, highlights), updates the B-spline curve, and triggers processing
        pipeline updates. Auto-scales related control points to maintain curve smoothness.
        
        Args:
            key (str): Control point identifier ('shadows', 'blacks', etc.)
            value (float): New value for the control point
        """
        if pref.verbose: print(" [CB] >> ToneCurveController.sliderChange(",key,",",value,")[callBackActive:",self.callBackActive,"] ")

        if self.callBackActive:
            newValues = self.model.setValue(key, value, autoScale = False)
            self.parent.controller.changeToneCurve(newValues) 
  
            points = self.model.evaluate()

            self.callBackActive =  False

            self.view.sliderShadows.setValue(int(newValues["shadows"][1]))
            self.view.editShadows.setText(str(newValues["shadows"][1]))

            self.view.sliderBlacks.setValue(int(newValues["blacks"][1]))
            self.view.editBlacks.setText(str(newValues["blacks"][1]))

            self.view.sliderMediums.setValue(int(newValues["mediums"][1]))
            self.view.editMediums.setText(str(newValues["mediums"][1]))

            self.view.sliderWhites.setValue(int(newValues["whites"][1]))
            self.view.editWhites.setText(str(newValues["whites"][1]))

            self.view.sliderHighlights.setValue(int(newValues["highlights"][1]))
            self.view.editHighlights.setText(str(newValues["highlights"][1]))

            self.callBackActive =  True
    # -----------------------------------------------------------------------------
    def setValues(self, valuesDict,callBackActive = False):
        """
        Set all tone curve control points programmatically.
        
        Updates all control points simultaneously and refreshes the interface
        to reflect the new curve shape. Typically used for loading saved
        curves or applying automatic curve calculations.
        
        Args:
            valuesDict (dict): Dictionary containing all control point values
                             with keys like 'shadows', 'blacks', 'mediums', etc.
            callBackActive (bool, optional): Whether to trigger callbacks (default: False)
        """
        if pref.verbose: print(" [CONTROL] >> ToneCurveController.setValue(",valuesDict,") ")

        self.callBackActive = callBackActive

        self.model.setValues(valuesDict)
        points = self.model.evaluate()

        self.view.sliderShadows.setValue(int(valuesDict["shadows"][1]))
        self.view.editShadows.setText(str(valuesDict["shadows"][1]))

        self.view.sliderBlacks.setValue(int(valuesDict["blacks"][1]))
        self.view.editBlacks.setText(str(valuesDict["blacks"][1]))

        self.view.sliderMediums.setValue(int(valuesDict["mediums"][1]))
        self.view.editMediums.setText(str(valuesDict["mediums"][1]))

        self.view.sliderWhites.setValue(int(valuesDict["whites"][1]))
        self.view.editWhites.setText(str(valuesDict["whites"][1]))

        self.view.sliderHighlights.setValue(int(valuesDict["highlights"][1]))
        self.view.editHighlights.setText(str(valuesDict["highlights"][1]))

        self.callBackActive = True
    # -----------------------------------------------------------------------------     
    # zj add for semi-auto curve begin
    def autoCurve(self):
        """
        Generate tone curve automatically using machine learning analysis.
        
        Analyzes the current image's histogram distribution and uses a trained
        neural network to predict optimal control point values for enhanced
        image appearance. The ML model considers luminance distribution patterns
        to suggest appropriate tone mapping.
        
        The process involves:
        1. Computing cumulative histogram of image luminance
        2. Feeding histogram to trained neural network
        3. Predicting optimal control point values
        4. Applying the generated curve to the processing pipeline
        """
        processPipe = self.parent.controller.model.getProcessPipe()
        if processPipe != None :
            idExposure = processPipe.getProcessNodeByName("tonecurve")
            bins = np.linspace(0,1,50+1)

            imageBeforeColorData = processPipe.processNodes[idExposure-1].outputImage.colorData
            imageBeforeColorData[imageBeforeColorData>1]=1
            imageBeforeY = colour.sRGB_to_XYZ(imageBeforeColorData, apply_cctf_decoding=False)[:,:,1]
            nphistBefore  = np.histogram(imageBeforeY, bins)[0]
            nphistBefore  = nphistBefore/np.amax(nphistBefore)

            npImgHistCumuNorm = np.empty_like(nphistBefore)
            npImgHistCumu = np.cumsum(nphistBefore)
            npImgHistCumuNorm = npImgHistCumu/np.max(npImgHistCumu)
            
            #predict keypoint value
            if self.networkModel == None:
                self.networkModel = Net(50,5)
                self.networkModel.load_state_dict(torch.load(self.weightFile))
                self.networkModel.eval()

            with torch.no_grad():
                x = Variable(torch.FloatTensor([npImgHistCumuNorm.tolist(),]), requires_grad=True)
                y_predict = self.networkModel(x)

            kpc = (y_predict[0]*100).tolist() 
            kpcDict = {'start':[0.0,0.0], 'shadows': [10.0,kpc[0]], 'blacks': [30.0,kpc[1]], 'mediums': [50.0,kpc[2]], 'whites': [70.0,kpc[3]], 'highlights': [90.0,kpc[4]], 'end': [100.0,100.0]}
            self.setValues(kpcDict,callBackActive = True)
            self.parent.controller.changeToneCurve(kpcDict) 
     
    # zj add for semi-auto curve end   

    # -----------------------------------------------------------------------------
    def reset(self, key):
        """
        Reset specific control point to its default value.
        
        Restores a single control point to its default position and updates
        the curve accordingly. Useful for correcting individual control
        points without affecting the entire curve.
        
        Args:
            key (str): Control point to reset ('shadows', 'blacks', etc.)
        """
        if pref.verbose: print(" [CONTROL] >> ToneCurveController.reset(",key,") ")

        valuesDefault = copy.deepcopy(self.model.default[key])[1]
        controls = self.model.setValue(key, valuesDefault)
        self.setValues(controls,callBackActive = False)

        self.parent.controller.changeToneCurve(controls) 
    # -----------------------------------------------------------------------------
    def plotCurve(self):
        """
        Display tone curve and histogram overlays.
        
        Renders the current tone curve along with optional histogram overlays
        for input, pre-processing, post-processing, and output stages. Provides
        visual feedback on how the curve affects the image's tonal distribution.
        
        The plot includes:
        - Grid lines for reference
        - Tone curve visualization
        - Control points markers
        - Optional histogram overlays for different processing stages
        """
        try:
            self.view.curve.plot([0,100],[0,100],'r--', clear=True)
            self.view.curve.plot([20,20],[0,100],'r--', clear=False)
            self.view.curve.plot([40,40],[0,100],'r--', clear=False)
            self.view.curve.plot([60,60],[0,100],'r--', clear=False)
            self.view.curve.plot([80,80],[0,100],'r--', clear=False)

            processPipe = self.parent.controller.model.getProcessPipe()
            idExposure = processPipe.getProcessNodeByName("tonecurve")

            bins = np.linspace(0,1,50+1)

            if self.showInput:
                imageInput = copy.deepcopy(processPipe.getInputImage())
                if imageInput.linear: imageInputColorData =colour.cctf_encoding(imageInput.colorData, function='sRGB')
                else: imageInputColorData = imageInput.colorData
                imageInputColorData[imageInputColorData>1]=1
                imageInputY = colour.sRGB_to_XYZ(imageInputColorData, apply_cctf_decoding=False)[:,:,1]
                nphistInput  = np.histogram(imageInputY, bins)[0]
                nphistInput  = nphistInput/np.amax(nphistInput)
                self.view.curve.plot(bins[:-1]*100,nphistInput*100,'k--',  clear=False)

            if self.showbefore:
                imageBeforeColorData = processPipe.processNodes[idExposure-1].outputImage.colorData
                imageBeforeColorData[imageBeforeColorData>1]=1
                imageBeforeY = colour.sRGB_to_XYZ(imageBeforeColorData, apply_cctf_decoding=False)[:,:,1]
                nphistBefore  = np.histogram(imageBeforeY, bins)[0]
                nphistBefore  = nphistBefore/np.amax(nphistBefore)
                self.view.curve.plot(bins[:-1]*100,nphistBefore*100,'b--',  clear=False)

            if self.showAfter:
                imageAftercolorData = processPipe.processNodes[idExposure].outputImage.colorData
                imageAftercolorData[imageAftercolorData>1]=1
                imageAfterY  = colour.sRGB_to_XYZ(imageAftercolorData,  apply_cctf_decoding=False)[:,:,1]
                nphistAfter   = np.histogram(imageAfterY, bins)[0]
                nphistAfter   =nphistAfter/np.amax(nphistAfter)
                self.view.curve.plot(bins[:-1]*100,nphistAfter*100,'b',     clear=False)

            if self.showOutput:
                imageAftercolorData = processPipe.getImage(toneMap=True).colorData
                imageAftercolorData[imageAftercolorData>1]=1
                imageAfterY  = colour.sRGB_to_XYZ(imageAftercolorData,  apply_cctf_decoding=False)[:,:,1]
                nphistAfter   = np.histogram(imageAfterY, bins)[0]
                nphistAfter   =nphistAfter/np.amax(nphistAfter)
                self.view.curve.plot(bins[:-1]*100,nphistAfter*100,'b',     clear=False)

            controlPointCoordinates= np.asarray(list(self.model.control.values()))
            self.view.curve.plot(controlPointCoordinates[1:-1,0],controlPointCoordinates[1:-1,1],'ro', clear=False)
            points = np.asarray(self.model.curve.evalpts)
            x = points[:,0]
            self.view.curve.plot(points[x<100,0],points[x<100,1],'r',clear=False)
        except:
            time.sleep(0.5)
            self.plotCurve()
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class LightnessMaskController():
    """
    Controller for lightness-based masking and tone range selection.
    
    This class manages the lightness mask interface, allowing users to selectively
    target specific tonal ranges (shadows, blacks, mediums, whites, highlights) 
    for localized editing operations. The masking system enables precise control
    over HDR processing by limiting effects to specific luminance ranges.
    
    Lightness masking is essential for HDR editing as it allows photographers
    to adjust different tonal ranges independently, creating more natural and
    visually pleasing results.
    
    Attributes:
        parent: Parent controller for coordination
        model (model.LightnessMaskModel): Lightness mask data model
        view (view.LightnessMaskView): Mask selection interface
        callBackActive (bool): Enable/disable callback processing
    
    Methods:
        maskChange: Handle mask enable/disable for tone ranges
        setValues: Set all mask states programmatically
    """
    def __init__(self, parent):
        """
        Initialize the lightness mask controller.
        
        Args:
            parent: Parent controller for coordination and processing pipeline access
        """
        if pref.verbose: print(" [CONTROL] >> MaskLightnessController.__init__(",")")

        self.parent= parent
        self.model = model.LightnessMaskModel(self)
        self.view = view.LightnessMaskView(self)

        self.callBackActive = True
    # -----------------------------------------------------------------------------
    def maskChange(self,key, on_off):
        """
        Handle mask enable/disable for specific tone ranges.
        
        Processes user interactions with mask checkboxes, updating the mask
        state for the specified tonal range and triggering processing updates
        to apply the mask changes.
        
        Args:
            key (str): Tone range identifier ('shadows', 'blacks', 'mediums', 
                      'whites', 'highlights')
            on_off (bool): Enable (True) or disable (False) the mask for this range
        """
        if pref.verbose: print(" [CB] >> MaskLightnessController.maskChange(",key,",",on_off,")[callBackActive:",self.callBackActive,"] ")

        maskState = self.model.maskChange(key, on_off)  
        self.parent.controller.changeLightnessMask(maskState) 
    # -----------------------------------------------------------------------------
    def setValues(self, values,callBackActive = False):
        """
        Set all mask states programmatically.
        
        Updates all tone range masks simultaneously and refreshes the interface
        to reflect the new mask configuration. Used for loading saved mask
        states or batch mask updates.
        
        Args:
            values (dict): Dictionary containing mask states for all tone ranges
                          with boolean values for each range
            callBackActive (bool, optional): Whether to trigger callbacks (default: False)
        """
        if pref.verbose: print(" [CONTROL] >> LightnessMaskController.setValue(",values,") ")

        self.callBackActive = callBackActive

        self.model.setValues(values)

        self.view.checkboxShadows.setChecked(values["shadows"])
        self.view.checkboxBlacks.setChecked(values["blacks"])
        self.view.checkboxMediums.setChecked(values["mediums"])
        self.view.checkboxWhites.setChecked(values["whites"])
        self.view.checkboxHighlights.setChecked(values["highlights"])   

        self.callBackActive = True
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class HDRviewerController():
    """
    Controller for external HDR display and viewer management.
    
    This class manages the external HDR display functionality, coordinating with
    HDR-capable monitors to show processed images at their full dynamic range.
    It handles display process management, image comparison modes, auto-preview
    functionality, and HDR display scaling.
    
    The HDR viewer enables real-time preview of processing results on calibrated
    HDR displays, providing immediate visual feedback for professional HDR
    image editing workflows.
    
    Attributes:
        parent: Parent controller for coordination
        model (model.HDRviewerModel): HDR display data model
        view: View component (set externally)
        viewerProcess: External HDR viewer process handle
    
    Methods:
        setView: Set the view component
        callBackUpdate: Update HDR display with current image
        callBackAuto: Toggle automatic HDR preview updates
        callBackCompare: Display before/after comparison
        displayFile: Show HDR file on external display
        displayIMG: Display processed image on HDR monitor
        displaySplash: Show default splash screen
        close: Terminate HDR viewer process
    """
    def __init__(self, parent):
        """
        Initialize the HDR viewer controller.
        
        Args:
            parent: Parent controller for coordination and image access
        """
        if pref.verbose: print(" [CONTROL] >> HDRviewerController.__init__(",")")

        self.parent= parent
        self.model = model.HDRviewerModel(self)
        self.view = None 

        self.viewerProcess = None

        self.displaySplash()

    def setView(self, view): 
        """
        Set the view component for the HDR viewer.
        
        Args:
            view: View component to associate with this controller
        """
        self.view = view

    def callBackUpdate(self):
        """
        Update HDR display with the current processed image.
        
        Retrieves the currently selected processing pipeline, processes the
        image without tone mapping (preserving HDR values), and displays
        the result on the HDR monitor. Updates the current image reference
        for comparison operations.
        """
        selectedProcessPipe = self.parent.view.imageGalleryController.model.getSelectedProcessPipe()
        img = selectedProcessPipe.getImage(toneMap = False)
        self.displayIMG(img)
        self.model.currentIMG = img

    def callBackAuto(self,on_off):
        """
        Toggle automatic HDR preview updates.
        
        Enables or disables automatic HDR display updates when processing
        parameters change, allowing for real-time preview on HDR monitors.
        
        Args:
            on_off (bool): Enable (True) or disable (False) auto-preview
        """
        self.parent.view.dock.view.childControllers[0].model.autoPreviewHDR = on_off

    def callBackCompare(self):
        """
        Display before/after comparison on HDR monitor.
        
        Creates a side-by-side comparison showing the previous processing
        result alongside the current result, enabling visual comparison
        of processing changes. If no previous result exists, updates
        with current processing.
        """
        if self.model.currentIMG:
            old = self.model.currentIMG
            old = old.process(hdrCore.processing.clip())

            sp = self.parent.view.imageGalleryController.model.getSelectedProcessPipe()
            img = sp.getImage(toneMap = False)
            img = img.process(hdrCore.processing.clip())


            h1, w1, _ = old.colorData.shape
            h2, w2, _ = img.colorData.shape
            hD, wD = self.model.displayModel['shape']
            hM = int((hD - max(h1,h2))/2)
            wM = int((wD - (w1+w2))/3)
            back =  np.ones((hD,wD,3))*0.2

            back[hM:hM+h1,wM:wM+w1,:] = old.colorData*self.model.displayModel['scaling']
            back[hM:hM+h2,2*wM+w1:2*wM+w1+w2,:] = img.colorData*self.model.displayModel['scaling']

            # save as temp.hdr
            colour.write_image(back,'temp.hdr', method='Imageio')
            self.displayFile('temp.hdr')

            self.model.currentIMG = img
        else: self.callBackUpdate()

    def displayFile(self, HDRfilename):
        """
        Display HDR file on external HDR monitor.
        
        Launches or updates the external HDR viewer process to display the
        specified HDR file. Handles process management and error recovery
        for reliable HDR display functionality.
        
        Args:
            HDRfilename (str): Path to HDR file to display
            
        Note:
            Uses HDRImageViewer.exe for Windows-based HDR display.
            Includes automatic process restart if initial launch fails.
        """

         # check that no current display process already open
        if self.viewerProcess:
            # the display HDR process is already running
            # close current
            subprocess.run(['taskkill', '/F', '/T', '/IM', "HDRImageViewer*"],capture_output=False)
            time.sleep(0.05)
        # run display HDR process
        self.viewerProcess = subprocess.Popen(["HDRImageViewer.exe","-f", "-input:"+HDRfilename, "-f", "-h"], shell=True)
        time.sleep(0.10)
        psData = subprocess.run(['tasklist'], capture_output=True, universal_newlines=True).stdout
        if not 'HDRImageViewer' in psData: 
            # re-run display HDR process
            self.viewerProcess = subprocess.Popen(["HDRImageViewer.exe","-f", "-input:"+HDRfilename, "-f", "-h"], shell=True)

    def displayIMG(self, img):
        """
        Display processed image on HDR monitor.
        
        Processes and displays a uHDR Image object on the external HDR monitor,
        applying appropriate clipping, scaling, and centering for optimal
        HDR display presentation.
        
        Args:
            img (hdrCore.image.Image): Processed HDR image to display
        """
        img = img.process(hdrCore.processing.clip())
        print(self.model.scaling())
        colorData = img.colorData*self.model.displayModel['scaling']

        h,w, _ = colorData.shape
        hD, wD = self.model.displayModel['shape']

        if w<wD:
            back = np.ones((hD,wD,3))*0.2
            marginW = int((wD-w)/2)
            marginH = int((hD-h)/2)

            back[marginH:marginH+h,marginW:marginW+w,:]=colorData

        # save as temp.hdr
        colour.write_image(back,'temp.hdr', method='Imageio')
        self.displayFile('temp.hdr')

    def displaySplash(self):
        """
        Display default splash screen on HDR monitor.
        
        Shows the default splash screen image, typically used when no
        specific image is being displayed or during application startup.
        Clears the current image reference.
        """
        self.model.currentIMG = None
        self.displayFile('grey.hdr')

    def close(self):
        """
        Close HDR viewer and terminate display process.
        
        Safely terminates the external HDR viewer process and cleans up
        resources. Should be called during application shutdown or when
        HDR display is no longer needed.
        """
        # check that no current display process already open
        if self.viewerProcess:
            # the display HDR process is already running
            # close current
            subprocess.run(['taskkill', '/F', '/T', '/IM', "HDRImageViewer*"],capture_output=False)
            self.viewerProcess = None
# ------------------------------------------------------------------------------------------
# ---- Class LchColorSelectorController ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class LchColorSelectorController:
    """
    Controller for LCH color space selective editing interface.
    
    This class manages the advanced color editing interface that operates in LCH
    (Lightness, Chroma, Hue) color space, providing precise control over color
    selection and modification. It enables photographers to selectively edit
    specific color ranges with professional-grade precision.
    
    The LCH color editor combines color selection tools with editing controls,
    allowing for targeted adjustments to specific colors while preserving the
    overall image integrity. This is essential for professional HDR color grading.
    
    Attributes:
        parent: Parent controller for coordination
        model (model.LchColorSelectorModel): LCH color editing data model
        view (view.LchColorSelectorView): LCH color editing interface
        idName (str): Unique identifier for this color editor instance
        callBackActive (bool): Enable/disable callback processing
    
    Methods:
        sliderHueChange: Handle hue range selection changes
        sliderChromaChange: Handle chroma range selection changes
        sliderLightnessChange: Handle lightness range selection changes
        sliderExposureChange: Handle exposure adjustment for selected colors
        sliderSaturationChange: Handle saturation adjustment for selected colors
        sliderContrastChange: Handle contrast adjustment for selected colors
        sliderHueShiftChange: Handle hue shift for selected colors
        checkboxMaskChange: Toggle mask visualization
        setValues: Set all color editing parameters programmatically
        resetSelection: Reset color selection to defaults
        resetEdit: Reset editing parameters to defaults
    """
    def __init__(self, parent, idName = None):
        """
        Initialize the LCH color selector controller.
        
        Args:
            parent: Parent controller for coordination
            idName (str, optional): Unique identifier for this editor instance
        """
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.__init__(",") ")
        self.parent = parent
        self.model =    model.LchColorSelectorModel(self)
        self.view =     view.LchColorSelectorView(self)

        self.idName = idName

        self.callBackActive = True

    def sliderHueChange(self, vMin, vMax):
        """
        Handle hue range selection changes.
        
        Args:
            vMin (float): Minimum hue value for selection (0-360 degrees)
            vMax (float): Maximum hue value for selection (0-360 degrees)
        """
        values  = self.model.setHueSelection(vMin,vMax)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderChromaChange(self, vMin, vMax):
        """
        Handle chroma range selection changes.
        
        Args:
            vMin (float): Minimum chroma value for selection (0-100)
            vMax (float): Maximum chroma value for selection (0-100)
        """
        values  = self.model.setChromaSelection(vMin,vMax)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderLightnessChange(self, vMin, vMax):
        """
        Handle lightness range selection changes.
        
        Args:
            vMin (float): Minimum lightness value for selection (0-100)
            vMax (float): Maximum lightness value for selection (0-100)
        """
        values  = self.model.setLightnessSelection(vMin,vMax)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderExposureChange(self, ev):
        """
        Handle exposure adjustment for selected colors.
        
        Args:
            ev (float): Exposure adjustment in EV stops for selected color range
        """
        values = self.model.setExposure(ev)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderSaturationChange(self, sat):
        """
        Handle saturation adjustment for selected colors.
        
        Args:
            sat (float): Saturation adjustment for selected color range
        """
        values = self.model.setSaturation(sat)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderContrastChange(self, cc):
        """
        Handle contrast adjustment for selected colors.
        
        Args:
            cc (float): Contrast adjustment for selected color range
        """
        values = self.model.setContrast(cc)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderHueShiftChange(self, hs):
        """
        Handle hue shift for selected colors.
        
        Args:
            hs (float): Hue shift amount in degrees for selected color range
        """
        values = self.model.setHueShift(hs)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def checkboxMaskChange(self,value): 
        """
        Toggle mask visualization for selected colors.
        
        Args:
            value (bool): Enable (True) or disable (False) mask visualization
        """
        values = self.model.setMask(value)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def setValues(self, values, callBackActive = False):
        """
        Set all color editing parameters programmatically.
        
        Updates all selection and editing controls simultaneously, typically
        used when loading saved color editing presets or initializing the
        interface with specific values.
        
        Args:
            values (dict): Complete color editing configuration including
                          selection ranges and editing parameters
            callBackActive (bool, optional): Whether to trigger callbacks (default: False)
        """
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.setValue(",values,") ")

        self.callBackActive = callBackActive
        # slider hue selection
        v = values['selection']['hue'] if 'hue' in values['selection'].keys() else (0,360)
        self.view.sliderHueMin.setValue(int(v[0]))
        self.view.sliderHueMax.setValue(int(v[1]))

        # slider chroma selection
        v = values['selection']['chroma'] if 'chroma' in values['selection'].keys() else (0,100)
        self.view.sliderChromaMin.setValue(int(v[0]))
        self.view.sliderChromaMax.setValue(int(v[1]))

        # slider lightness
        v = values['selection']['lightness'] if 'lightness' in values['selection'].keys() else (0,100)
        self.view.sliderLightMin.setValue(int(v[0]*3))
        self.view.sliderLightMax.setValue(int(v[1]*3))

        # hue shift editor
        v = values['edit']['hue'] if 'hue' in values['edit'].keys() else 0
        self.view.sliderHueShift.setValue(int(v))
        self.view.valueHueShift.setText(str(v)) 

        # exposure editor
        v : int = values['edit']['exposure'] if 'exposure' in values['edit'].keys() else 0
        self.view.sliderExposure.setValue(int(v*30))
        self.view.valueExposure.setText(str(v)) 

        # contrast editor
        v : int = values['edit']['contrast'] if 'contrast' in values['edit'].keys() else 0
        self.view.sliderContrast.setValue(int(v))
        self.view.valueContrast.setText(str(v))  

        # saturation editor
        v : int = values['edit']['saturation'] if 'saturation' in values['edit'].keys() else 0
        self.view.sliderSaturation.setValue(int(v))
        self.view.valueSaturation.setText(str(v))  

        # mask
        v : bool = values['mask'] if 'mask' in values.keys() else False
        self.view.checkboxMask.setChecked(values['mask'])             

        self.model.setValues(values)

        self.callBackActive = True

    # -----
    def resetSelection(self): 
        """
        Reset color selection parameters to defaults.
        
        Restores the color selection range (hue, chroma, lightness) to
        default values while preserving editing parameters. Useful for
        starting fresh color selections.
        """
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.resetSelection(",") ")

        default = copy.deepcopy(self.model.default)
        current = copy.deepcopy(self.model.getValues())
        
        current['selection'] = default['selection']

        self.setValues(current,callBackActive = True)
        self.callBackActive = True

    def resetEdit(self): 
        """
        Reset editing parameters to defaults.
        
        Restores all editing adjustments (exposure, saturation, contrast,
        hue shift) to default values while preserving color selection.
        Useful for clearing applied adjustments.
        """
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.resetEdit(",") ")

        default = copy.deepcopy(self.model.default)
        current = copy.deepcopy(self.model.getValues())
        
        current['edit'] = default['edit']

        self.setValues(current,callBackActive = True)
        self.callBackActive = True
# ------------------------------------------------------------------------------------------
# ---- Class LchColorSelectorController ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class GeometryController:
    """
    Controller for geometric transformations and image adjustments.
    
    This class manages geometric processing controls including cropping adjustments,
    rotation, and spatial positioning. It provides essential tools for correcting
    perspective issues, adjusting composition, and applying geometric corrections
    to HDR images while preserving image quality.
    
    The geometry controller handles precision adjustments that are critical for
    professional HDR photography, especially when correcting architectural
    photography or adjusting horizon lines.
    
    Attributes:
        parent: Parent controller for coordination
        model (model.GeometryModel): Geometric transformation data model
        view (view.GeometryView): Geometry adjustment interface
        callBackActive (bool): Enable/disable callback processing
    
    Methods:
        sliderCroppingVerticalAdjustementChange: Handle vertical crop position
        sliderRotationChange: Handle rotation angle adjustments
        setValues: Set all geometry parameters programmatically
    """
    def __init__(self, parent ):
        """
        Initialize the geometry controller.
        
        Args:
            parent: Parent controller for coordination and processing pipeline access
        """
        if pref.verbose: print(" [CONTROL] >> GeometryController.__init__(",") ")
        self.parent = parent
        self.model =    model.GeometryModel(self)
        self.view =     view.GeometryView(self)

        self.callBackActive = True
    # callbacks
    def sliderCroppingVerticalAdjustementChange(self,v):
        """
        Handle vertical cropping position adjustments.
        
        Adjusts the vertical position of the crop area, allowing users to
        reframe the image vertically while maintaining the aspect ratio.
        
        Args:
            v (float): Vertical adjustment value (typically -100 to +100)
        """
        values = self.model.setCroppingVerticalAdjustement(v)
        if self.callBackActive : self.parent.controller.changeGeometry(values)

    def sliderRotationChange(self,v):
        """
        Handle rotation angle adjustments.
        
        Adjusts the image rotation angle for correcting tilted horizons or
        perspective issues. Rotation is applied with automatic cropping to
        maintain rectangular output.
        
        Args:
            v (float): Rotation angle in degrees
        """
        values = self.model.setRotation(v)
        if self.callBackActive : self.parent.controller.changeGeometry(values)

    def setValues(self, values, callBackActive = False):
        """
        Set all geometry parameters programmatically.
        
        Updates all geometric transformation controls simultaneously, typically
        used when loading saved geometry settings or batch updates.
        
        Args:
            values (dict): Dictionary containing geometry parameters including
                          'up' (vertical adjustment) and 'rotation' (angle)
            callBackActive (bool, optional): Whether to trigger callbacks (default: False)
        """
        if pref.verbose: print(" [CONTROL] >> GeometryController.setValue(",values,") ")

        up =        values['up']        if 'up' in values.keys()        else 0.0
        rotation =  values['rotation']  if 'rotation' in values.keys()  else 0.0

        self.callBackActive = callBackActive

        self.view.sliderCroppingVerticalAdjustement.setValue(int(up))
        self.view.sliderRotation.setValue(int(rotation*6))
       
        self.model.setValues(values)

        self.callBackActive = True
# ------------------------------------------------------------------------------------------
# ---- Class AestheticsImageController -----------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageAestheticsController:
    """
    Controller for image aesthetics analysis and display.
    
    This class manages the MIAM (Machine Intelligence for Aesthetic Metrics)
    interface, providing comprehensive aesthetic analysis of HDR images including
    quality scoring, color palette extraction, and aesthetic metrics visualization.
    
    The aesthetics controller integrates computer vision algorithms to evaluate
    image quality across multiple dimensions, helping photographers understand
    the aesthetic properties of their HDR images and make informed editing decisions.
    
    Attributes:
        parent: Parent controller for coordination
        model (model.ImageAestheticsModel): Aesthetics analysis data model
        view (view.ImageAestheticsView): Aesthetics display interface
    
    Methods:
        buildView: Build or rebuild the aesthetics interface
        setProcessPipe: Set processing pipeline for aesthetic analysis
    """
    def __init__(self, parent=None, HDRcontroller = None):
        """
        Initialize the image aesthetics controller.
        
        Args:
            parent: Parent controller for coordination
            HDRcontroller: HDR display controller (not used in current implementation)
        """
        if pref.verbose: print(" [CONTROL] >> AestheticsImageController.__init__(",")")

        self.parent = parent
        self.model = model.ImageAestheticsModel(self)
        self.view = view.ImageAestheticsView(self)
    # --------------------------------------------------------------------------------------
    def buildView(self,processPipe=None):
        """
        Build or rebuild the aesthetics interface view.
        
        Reconstructs the aesthetics analysis interface, typically called when
        switching to the aesthetics panel or refreshing the analysis display.
        
        Args:
            processPipe: Optional processing pipeline to analyze during build
        """
        if pref.verbose: print(" [CONTROL] >> AestheticsImageController.buildView()")

        # called when MultiDockController recall a controller/view 
        self.view = view.ImageAestheticsView(self)
        if processPipe: self.setProcessPipe(processPipe)
    # --------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        """
        Set processing pipeline for aesthetic analysis.
        
        Configures the aesthetics interface to analyze the specified processing
        pipeline's image, computing aesthetic metrics and generating visual
        analysis including color palette extraction.
        
        Args:
            processPipe: Processing pipeline containing image for analysis
            
        Returns:
            bool: True (always successful for aesthetics analysis)
        """
        if pref.verbose: print(" [CONTROL] >> AestheticsImageController.setProcessPipe()")

        self.model.setProcessPipe(processPipe)
        self.view.setProcessPipe(processPipe, self.model.getPaletteImage())

        return True
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# --- message widget functions -------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def messageBox(title, text):
    """
    Display informational message dialog.
    
    Creates and displays a simple message box with OK button for user notifications.
    
    Args:
        title (str): Dialog window title
        text (str): Message text to display
    """
    msg = QMessageBox()
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setEscapeButton(QMessageBox.Close)
    msg.exec_()
# -----------------------------------------------------------------------------
def okCancelBox(title, text):
    """
    Display confirmation dialog with OK/Cancel options.
    
    Creates and displays a confirmation dialog allowing user to proceed or cancel.
    
    Args:
        title (str): Dialog window title
        text (str): Confirmation message text
        
    Returns:
        int: User's choice (QMessageBox.Ok or QMessageBox.Cancel)
    """
    msg = QMessageBox()
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.setEscapeButton(QMessageBox.Close)
    return msg.exec_()
# ------------------------------------------------------------------------------------------
# ---- Class ColorEditorsAutoController ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class  ColorEditorsAutoController:
    """
    Controller for automatic color editor configuration and management.
    
    This class provides automated color editing capabilities by analyzing image
    content and automatically configuring multiple color editor instances with
    optimal settings. It uses advanced algorithms to identify dominant colors
    and suggest appropriate editing parameters for professional color grading.
    
    The automatic color editor system streamlines the workflow for complex color
    corrections by providing intelligent starting points for manual fine-tuning.
    
    Attributes:
        parent: Parent controller for coordination
        controlled (list): List of controlled color editor instances
        stepName (str): Processing step name for this auto-controller
        model (model.ColorEditorsAutoModel): Auto-color editing data model
        view (view.ColorEditorsAutoView): Auto-color editing interface
        callBackActive (bool): Enable/disable callback processing
    
    Methods:
        auto: Trigger automatic color editor configuration
    """
    def __init__(self, parent, controlledColorEditors, stepName ):
        """
        Initialize the automatic color editors controller.
        
        Args:
            parent: Parent controller for coordination
            controlledColorEditors (list): List of color editor controllers to manage
            stepName (str): Processing step identifier for this auto-controller
        """
        if pref.verbose: print(" [CONTROL] >> ColorEditorsAutoController.__init__(",") ")

        self.parent = parent
        self.controlled = controlledColorEditors
        self.stepName =stepName

        self.model =    model.ColorEditorsAutoModel(self, stepName,len(controlledColorEditors), removeBlack= True)
        self.view =     view.ColorEditorsAutoView(self)

        self.callBackActive = True
    # callbacks
    def auto(self): 
        """
        Trigger automatic color editor configuration.
        
        Analyzes the current image to identify dominant colors and automatically
        configures all controlled color editors with appropriate selection ranges
        and initial editing parameters. Resets all editors before applying new
        automatic configurations.
        """
        if pref.verbose: print(" [CONTROL] >> ColorEditorsAutoController.auto(",") ")
        for ce in self.controlled: ce.resetSelection(); ce.resetEdit()
        values = self.model.compute()

        if values != None:
            for i,v in enumerate(values): self.controlled[i].setValues(v, callBackActive = False)

