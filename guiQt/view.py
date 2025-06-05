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
guiQt.view module: PyQt5 view components for uHDR GUI application.

This module implements the View component of the Model-View-Controller (MVC)
architecture pattern used in uHDR. It contains PyQt5 widgets and interfaces
that handle user interaction and visual presentation of HDR image editing tools.

The module provides:
- Main application window with menu system and docking
- Image gallery with grid layout and pagination  
- HDR image editing controls and parameter adjustment
- Tone curve editing with B-spline visualization
- Color space editing in LCH coordinates
- Lightness masking for selective adjustments
- Geometry transformation controls
- Image information and metadata display
- Aesthetics analysis visualization
- HDR display and external monitor support

Key View Classes:
    - AppView: Main application window with menus and docking
    - ImageGalleryView: Grid-based image gallery with pagination
    - EditImageView: Complete HDR editing interface
    - ImageInfoView: Image metadata and information display
    - ToneCurveView: B-spline tone curve editing with matplotlib
    - LchColorSelectorView: LCH color space selection interface
    - HDRviewerView: HDR display controls and preview
    - ImageAestheticsView: Color palette and aesthetics visualization

Widget Utilities:
    - ImageWidgetView: Basic image display widget
    - FigureWidget: Matplotlib integration for curve editing
    - AdvanceSliderView: Enhanced slider with auto/reset functionality
    - AdvanceLineEdit: Labeled line edit widget
    - AdvanceCheckBox: Labeled checkbox widget

The views handle user interaction events and delegate business logic
to their associated controllers while providing visual feedback and
real-time parameter adjustment capabilities.
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow, QSplitter, QFrame, QDockWidget, QDesktopWidget
from PyQt5.QtWidgets import QSplitter, QFrame, QSlider, QCheckBox, QGroupBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QLayout, QScrollArea, QFormLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit,QLineEdit, QComboBox, QSpinBox
from PyQt5.QtWidgets import QAction, QProgressBar, QDialog
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtWidgets 

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from datetime import datetime
import time

import numpy as np
import hdrCore.image, hdrCore.processing
import math, enum
import functools

from . import controller, model
import hdrCore.metadata
import preferences.preferences as pref

# ------------------------------------------------------------------------------------------
# --- class ImageWidgetView(QWidget) -------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageWidgetView(QWidget):
    """
    Basic image display widget for showing numpy arrays or HDR images.
    
    Provides a simple image viewer widget that can display color data
    as QPixmap with automatic scaling and aspect ratio preservation.
    Handles clipping and conversion from floating-point HDR data to
    8-bit RGB for Qt display.
    
    Attributes:
        - controller: Reference to controlling ImageWidgetController
        - label (QLabel): Qt label widget for pixmap display
        - imagePixmap (QPixmap): Current pixmap for display
    """

    def __init__(self,controller,colorData = None):
        """
        Initialize image widget view.
        
        Args:
            controller: Parent ImageWidgetController instance
            colorData (numpy.ndarray, optional): Initial image data to display
        """
        super().__init__()
        self.controller = controller
        self.label = QLabel(self)   # create a QtLabel for pixmap
        if not isinstance(colorData, np.ndarray): colorData = ImageWidgetView.emptyImageColorData()
        # self.colorData = colorData  # image content attributes           
        self.setPixmap(colorData)  

    def resize(self):
        """
        Update widget and pixmap scaling to current size.
        
        Resizes the internal label and scales the pixmap to fit the widget
        while preserving aspect ratio.
        """
        self.label.resize(self.size())
        self.label.setPixmap(self.imagePixmap.scaled(self.size(),Qt.KeepAspectRatio))

    def resizeEvent(self,event):
        """
        Handle Qt resize events.
        
        Args:
            event: Qt resize event object
        """
        self.resize()
        super().resizeEvent(event)

    def setPixmap(self,colorData):
        """
        Set image from numpy array with HDR to sRGB conversion.
        
        Converts floating-point HDR image data to 8-bit RGB for Qt display.
        Applies clipping to [0,1] range and handles format conversion.
        
        Args:
            colorData (numpy.ndarray): RGB image data (height, width, 3)
            
        Returns:
            QPixmap: Generated pixmap for display
        """
        if not isinstance(colorData, np.ndarray): 
            colorData = ImageWidgetView.emptyImageColorData()
        # self.colorData = colorData
        height, width, channel = colorData.shape   # compute pixmap
        bytesPerLine = channel * width
        # clip
        colorData[colorData>1.0] = 1.0
        colorData[colorData<0.0] = 0.0

        qImg = QImage((colorData*255).astype(np.uint8), width, height, bytesPerLine, QImage.Format_RGB888) # QImage
        self.imagePixmap = QPixmap.fromImage(qImg)
        self.resize()

        return self.imagePixmap

    def setQPixmap(self, qPixmap):
        """
        Set pre-processed QPixmap directly.
        
        Args:
            qPixmap (QPixmap): Pre-processed pixmap for display
        """
        self.imagePixmap = qPixmap
        self.resize()

    def emptyImageColorData(): 
        """
        Generate default placeholder image for empty display.
        
        Returns:
            numpy.ndarray: Gray placeholder image (90x160x3) 
        """
        return np.ones((90,160,3))*(220/255) 
# ------------------------------------------------------------------------------------------
# --- class FigureWidget(FigureCanvas ------------------------------------------------------
# ------------------------------------------------------------------------------------------
class FigureWidget(FigureCanvas):
    """
    Matplotlib figure widget for embedding plots in Qt interface.
    
    Provides matplotlib integration for displaying curves, histograms,
    and other plots within the Qt GUI. Used primarily for tone curve
    visualization and editing.
    
    Attributes:
        - fig (Figure): Matplotlib figure object
        - axes: Matplotlib axes for plotting
    """

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        """
        Initialize matplotlib figure widget.
        
        Args:
            parent: Parent Qt widget
            width (int): Figure width in inches
            height (int): Figure height in inches
            dpi (int): Figure resolution in dots per inch
        """
        # create Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)    # explicite call of super constructor
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
        self.setMinimumSize(200, 200)

    def plot(self,X,Y,mode, clear=False):
        """
        Draw line plot with specified data and style.
        
        Args:
            X (array-like): X-coordinate data points
            Y (array-like): Y-coordinate data points
            mode (str): Matplotlib line style (e.g., 'r--', 'b-')
            clear (bool): Clear previous plots before drawing
        """
        if clear: self.axes.clear()
        self.axes.plot(X,Y,mode)
        try:
            self.fig.canvas.draw()
        except Exception:
            time.sleep(0.5)
            self.fig.canvas.draw()

# ------------------------------------------------------------------------------------------
# --- class ImageGalleryView(QSplitter) ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageGalleryView(QSplitter):
    """
    Grid-based image gallery with pagination and navigation controls.
    
    Displays multiple images in a configurable grid layout with pagination
    support for large image collections. Provides interactive navigation
    buttons and supports different grid configurations (1x1, 3x2, 6x4, 9x6).
    
    Layout Structure:
        +-------------------------------------------+
        | +----+ +----+ +----+ +----+ +----+ +----+ | \
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |   >   GridLayout
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ | /
        +-------------------------------------------+  <    splitter
        | [<] [1x1][3x2][6x4][9x6][page number] [>] |       [pushButton] HorizontalLayout
        +-------------------------------------------+
    
    Attributes:
        - controller: Reference to ImageGalleryController
        - shapeMode: Current grid layout mode (GalleryMode enum)
        - pageNumber (int): Current page index (0-based)
        - imagesControllers (list): List of ImageWidgetController instances
        - images (QFrame): Frame containing image grid
        - imagesLayout (QGridLayout): Grid layout for images
        - buttons (QWidget): Navigation button container
        - pageNumberLabel (QLabel): Current page display
    """
    def __init__(self,controller_=None,shapeMode=None):
        """
        Initialize image gallery view.
        
        Args:
            controller_: Parent ImageGalleryController instance
            shapeMode: Initial grid layout mode (defaults to 3x2)
        """
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.__init__(",")")

        super().__init__(Qt.Vertical)

        self.controller= controller_
        self.shapeMode = controller.GalleryMode._3x2 if not shapeMode else shapeMode    # default display mode 
        self.pageNumber =0

        self.imagesControllers = []

        self.images = QFrame()
        self.images.setFrameShape(QFrame.StyledPanel)
        self.imagesLayout = QGridLayout()
        self.images.setLayout(self.imagesLayout)

        self.buildGridLayoutWidgets()

        self.previousPageButton =   QPushButton('<')
        self.previousPageButton.clicked.connect(self.controller.callBackButton_previousPage)
        self._1x1Button =           QPushButton('1x1')
        self._1x1Button.clicked.connect(self.controller.callBackButton_1x1)
        self._2x1Button =           QPushButton('2x1')
        self._2x1Button.clicked.connect(self.controller.callBackButton_2x1)
        self._3x2Button =           QPushButton('3x2')
        self._3x2Button.clicked.connect(self.controller.callBackButton_3x2)
        self._6x4Button =           QPushButton('6x4')
        self._6x4Button.clicked.connect(self.controller.callBackButton_6x4)
        self._9x6Button =           QPushButton('9x6')
        self._9x6Button.clicked.connect(self.controller.callBackButton_9x6)
        self.nextPageButton =       QPushButton('>')
        self.nextPageButton.clicked.connect(self.controller.callBackButton_nextPage)

        self.pageNumberLabel = QLabel(str(self.pageNumber)+"/ ...")

        self.buttons = QWidget()
        self.buttonsLayout = QHBoxLayout()
        self.buttons.setLayout(self.buttonsLayout)
        self.buttonsLayout.addWidget(self.previousPageButton)
        self.buttonsLayout.addWidget(self._1x1Button)
        self.buttonsLayout.addWidget(self._2x1Button)
        self.buttonsLayout.addWidget(self._3x2Button)
        self.buttonsLayout.addWidget(self._6x4Button)
        self.buttonsLayout.addWidget(self._9x6Button)
        self.buttonsLayout.addWidget(self.nextPageButton)

        self.buttonsLayout.addWidget(self.pageNumberLabel)

        self.addWidget(self.images)
        self.addWidget(self.buttons)
        self.setSizes([1525,82])

    def currentPage(self): 
        """
        Get current page number.
        
        Returns:
            int: Current page index (0-based)
        """
        return self.pageNumber

    def changePageNumber(self,step):
        """
        Navigate to different page with wraparound.
        
        Changes the current page by the specified step, with automatic
        wraparound at the beginning and end of the page range.
        
        Args:
            step (int): Page increment/decrement (+1 for next, -1 for previous)
        """
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.changePageNumber(",step,")")

        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode)*controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes)-1)//nbImagePerPage) + 1

        if len(self.controller.model.processPipes) > 0 :

            oldPageNumber = self.pageNumber
            if (self.pageNumber+step) > maxPage-1:
                self.pageNumber = 0
            elif (self.pageNumber+step) <0:
                self.pageNumber = maxPage-1
            else:
                self.pageNumber  = self.pageNumber+step
            self.updateImages()
            self.controller.model.loadPage(self.pageNumber)
            if pref.verbose: print(" [VIEW] >> ImageGalleryView.changePageNumber(currentPage:",self.pageNumber," | max page:",maxPage,")")

    def updateImages(self):
        """
        Refresh image display for current page.
        
        Updates all image widgets in the grid with placeholder content
        and refreshes the page number display. Actual images are loaded
        asynchronously by the model.
        """
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.updateImages(",")")

        """ update images content """
        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode)*controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes)-1)//nbImagePerPage) + 1
        
        index=0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)): 
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                # get image controllers                                                                                         
                iwc = self.imagesControllers[index]
                iwc.view.setPixmap(ImageWidgetView.emptyImageColorData())                                                 
                index +=1                                                                                                                                                                                                           
        self.pageNumberLabel.setText(str(self.pageNumber)+"/"+str(maxPage-1))

    def updateImage(self, idx, processPipe, filename):
        """
        Update specific image in grid with loaded content.
        
        Args:
            idx (int): Index of image within current page
            processPipe (ProcessPipe): Loaded image processing pipeline
            filename (str): Image filename for status display
        """
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.updateImage(",")")
        imageWidgetController = self.imagesControllers[idx]                                 
        imageWidgetController.setImage(processPipe.getImage())
        self.controller.parent.statusBar().showMessage("loading of image "+filename+" done!")

    def resetGridLayoutWidgets(self):
        """
        Clear current grid layout widgets.
        
        Removes all image widgets from the grid layout and cleans up
        memory by deleting widget references.
        """
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.resetGridLayoutWidgets(",")")

        for w in self.imagesControllers:
            self.imagesLayout.removeWidget(w.view)
            w.view.deleteLater()
        self.imagesControllers = []

    def buildGridLayoutWidgets(self):
        """
        Create new grid layout widgets based on current shape mode.
        
        Generates ImageWidgetController instances for each position in
        the grid and adds them to the layout with proper positioning.
        """
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.buildGridLayoutWidgets(",")")

        imageIndex = 0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)): 
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                iwc = controller.ImageWidgetController(id=imageIndex)
                self.imagesControllers.append(iwc)
                self.imagesLayout.addWidget(iwc.view,i,j)
                imageIndex +=1

    def wheelEvent(self, event):
        """
        Handle mouse wheel navigation between pages.
        
        Args:
            event: Qt wheel event with scroll direction
        """
        if pref.verbose: print(" [EVENT] >> ImageGalleryView.wheelEvent(",")")

        if event.angleDelta().y() < 0 :     
            self.changePageNumber(+1)
            if self.shapeMode == controller.GalleryMode._1x1 : 
                self.controller.selectImage(0)

        else :                              
            self.changePageNumber(-1)
            if self.shapeMode == controller.GalleryMode._1x1 : 
                self.controller.selectImage(0)
        event.accept()

    def mousePressEvent(self,event):
        """
        Handle image selection clicks.
        
        Determines which image was clicked based on mouse position
        and notifies the controller for selection handling.
        
        Args:
            event: Qt mouse press event with position
        """
        if pref.verbose: print(" [EVENT] >> ImageGalleryView.mousePressEvent(",")")

        # self.childAt(event.pos()) return QLabel .parent() should be ImageWidget object
        if isinstance(self.childAt(event.pos()).parent(),ImageWidgetView):
            id = self.childAt(event.pos()).parent().controller.id()
        else:
            id = -1

        if id != -1: # an image is clicked select it!
            self.controller.selectImage(id)
        event.accept()
# ------------------------------------------------------------------------------------------
# --- class AppView(QMainWindow) -----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AppView(QMainWindow):
    """
    Main application window for uHDR HDR image editing software.
    
    Provides the primary user interface for uHDR with comprehensive menu system,
    dockable panels, and central image gallery. Manages window geometry, screen
    configuration, and application-wide functionality including file operations,
    HDR display management, and export capabilities.
    
    Window Structure:
    - Central Widget: Image gallery with pagination
    - Right Dock: Multi-panel interface (Edit/Info/Aesthetics)
    - Menu Bar: File, Display HDR, Export, Dock, Preferences
    - Status Bar: Real-time operation feedback
    
    Attributes:
        - controller (AppController): Parent application controller
        - imageGalleryController (ImageGalleryController): Central gallery management
        - dock (MultiDockController): Right panel controller
        - topContainer (QWidget): Central widget container
        - menuExport (QAction): Export menu action reference
        - menuExportAll (QAction): Export all menu action reference
    """
    def __init__(self, _controller = None, shapeMode=None, HDRcontroller=None):
        """
        Initialize main application window.
        
        Args:
            _controller (AppController): Parent application controller
            shapeMode: Initial gallery layout mode
            HDRcontroller: HDR display controller reference
        """
        super().__init__()
        # --------------------
        scale = 0.8
        # --------------------   
        # attributes
        self.controller = _controller
        self.setWindowGeometry(scale=scale)
        self.setWindowTitle('uHDR - Rémi Cozot (c) 2020-2021')      # title  
        self.statusBar().showMessage('Welcome to uHDR!')         # status bar

        self.topContainer = QWidget()
        self.topLayout = QHBoxLayout()

        self.imageGalleryController = controller.ImageGalleryController(self)

        self.topLayout.addWidget(self.imageGalleryController.view)

        self.topContainer.setLayout(self.topLayout)
        self.setCentralWidget(self.topContainer)
        # ----------------------------------
        self.dock = controller.MultiDockController(self, HDRcontroller)
        self.addDockWidget(Qt.RightDockWidgetArea,self.dock.view)
        self.resizeDocks([self.dock.view],[int(self.controller.screenSize[0].width()*scale//4)],Qt.Horizontal)

        # ----------------------------------
        # build menu
        self.buildFileMenu()
        self.buildDockMenu()
        self.buildDisplayHDR()
        self.buildExport()
        self.buildPreferences()
    # ------------------------------------------------------------------------------------------
    def getImageGalleryController(self): 
        """
        Get reference to the central image gallery controller.
        
        Returns:
            ImageGalleryController: Central gallery management controller
        """
        return self.imageGalleryController
    # ------------------------------------------------------------------------------------------
    def resizeEvent(self, event): 
        """
        Handle Qt window resize events.
        
        Args:
            event: Qt resize event object
        """
        super().resizeEvent(event)
    # ------------------------------------------------------------------------------------------
    def setWindowGeometry(self, scale=0.8):
        """
        Configure window size and position based on available displays.
        
        Automatically detects multiple displays and positions window on
        secondary display if available, otherwise uses primary display.
        
        Args:
            scale (float): Window scaling factor (default 0.8)
        """
        displayCoord = QDesktopWidget().screenGeometry(1)
        if len(self.controller.screenSize) > 1:
            width, height = self.controller.screenSize[1].width(), self.controller.screenSize[1].height()
        else:
            width, height = self.controller.screenSize[0].width(), self.controller.screenSize[0].height()

        self.setGeometry(displayCoord.left(), displayCoord.top()+50, math.floor(width*scale), math.floor(height*scale))
        self.showMaximized()
    # ------------------------------------------------------------------------------------------
    def buildFileMenu(self):
        """
        Create file operations menu with directory selection, save, and quit.
        
        Menu Items:
        - Select directory (Ctrl+O): Choose working directory
        - Save (Ctrl+S): Save ProcessPipe metadata 
        - Quit (Ctrl+Q): Exit application with cleanup
        """
        menubar = self.menuBar()# get menubar
        fileMenu = menubar.addMenu('&File')# file menu

        selectDir = QAction('&Select directory', self)        
        selectDir.setShortcut('Ctrl+O')
        selectDir.setStatusTip('[File] select a directory')
        selectDir.triggered.connect(self.controller.callBackSelectDir)
        fileMenu.addAction(selectDir)

        selectSave = QAction('&Save', self)        
        selectSave.setShortcut('Ctrl+S')
        selectSave.setStatusTip('[File] saving processpipe metadata')
        selectSave.triggered.connect(self.controller.callBackSave)
        fileMenu.addAction(selectSave)

        quit = QAction('&Quit', self)        
        quit.setShortcut('Ctrl+Q')
        quit.setStatusTip('[File] saving updates and quit')
        quit.triggered.connect(self.controller.callBackQuit)
        fileMenu.addAction(quit)
    # ------------------------------------------------------------------------------------------    
    def buildPreferences(self):
        """
        Create HDR display preferences menu with dynamic display selection.
        
        Automatically generates menu items for each configured HDR display
        from preferences, allowing runtime switching between different HDR
        monitor configurations.
        """

        menubar = self.menuBar()# get menubar
        prefMenu = menubar.addMenu('&Preferences')# file menu

        displayList = pref.getHDRdisplays().keys()

        # function for callback
        def cbd(tag): 
            pref.setHDRdisplay(tag)
            self.statusBar().showMessage("swithcing HDR Display to: "+tag+"!")
            self.menuExport.setText('&Export to '+pref.getHDRdisplay()['tag'])
            self.menuExportAll.setText('&Export All to '+pref.getHDRdisplay()['tag'])        

        prefDisplays = []
        for i,d in enumerate(displayList):
            if d != 'none':
                prefDisplay = QAction('&Set display to '+d, self) 
                p_cbd = functools.partial(cbd, d)
                prefDisplay.triggered.connect(p_cbd)
                prefMenu.addAction(prefDisplay)

    # ------------------------------------------------------------------------------------------
    def buildDisplayHDR(self):
        """
        Create HDR display menu for real-time preview and comparison.
        
        Menu Items:
        - Display HDR image (Ctrl+H): Show on HDR monitor
        - Compare raw and edited (Ctrl+C): Side-by-side comparison
        - Reset HDR display (Ctrl+K): Clear HDR preview
        """
        menubar = self.menuBar()# get menubar
        displayHDRmenu = menubar.addMenu('&Display HDR')# file menu

        displayHDR = QAction('&Display HDR image', self)        
        displayHDR.setShortcut('Ctrl+H')
        displayHDR.setStatusTip('[Display HDR] display HDR image')
        displayHDR.triggered.connect(self.controller.callBackDisplayHDR)
        displayHDRmenu.addAction(displayHDR)
        # ------------------------------------
        displayHDR = QAction('&Compare raw and edited HDR image', self)        
        displayHDR.setShortcut('Ctrl+C')
        displayHDR.setStatusTip('[Display HDR] compare raw HDR image and edited one')
        displayHDR.triggered.connect(self.controller.callBackCompareRawEditedHDR)
        displayHDRmenu.addAction(displayHDR)
        # ------------------------------------
        closeHDR = QAction('&reset HDR display', self)        
        closeHDR.setShortcut('Ctrl+K')
        closeHDR.setStatusTip('[Display HDR] reset HDR window')
        closeHDR.triggered.connect(self.controller.callBackCloseDisplayHDR)
        displayHDRmenu.addAction(closeHDR)
    # ------------------------------------------------------------------------------------------  
    def buildExport(self):
        """
        Create export menu for saving processed HDR images.
        
        Menu Items:
        - Export to [display] (Ctrl+X): Export current image for HDR display
        - Export All to [display] (Ctrl+Y): Batch export all images
        
        Display target automatically updates based on preferences selection.
        """
        menubar = self.menuBar()# get menubar
        exportHDR = menubar.addMenu('&Export HDR image')# file menu

        self.menuExport = QAction('&Export to '+pref.getHDRdisplay()['tag'], self)        
        self.menuExport.setShortcut('Ctrl+X')
        self.menuExport.setStatusTip('[Export HDR image] save HDR image file for HDR display')
        self.menuExport.triggered.connect(self.controller.callBackExportHDR)
        exportHDR.addAction(self.menuExport)

        self.menuExportAll = QAction('&Export All to '+pref.getHDRdisplay()['tag'], self)        
        self.menuExportAll.setShortcut('Ctrl+Y')
        self.menuExportAll.setStatusTip('[Export all HDR images] save HDR image files for HDR display.')
        self.menuExportAll.triggered.connect(self.controller.callBackExportAllHDR)
        exportHDR.addAction(self.menuExportAll)
    # ------------------------------------------------------------------------------------------        
    def buildDockMenu(self):
        """
        Create dock panel switching menu for right-side interface.
        
        Menu Items:
        - Info and Metadata (Ctrl+I): Image information panel
        - Edit (Ctrl+E): HDR editing controls panel  
        - Image Aesthetics (Ctrl+A): Aesthetics analysis panel
        """
        menubar = self.menuBar()# get menubar
        dockMenu = menubar.addMenu('&Dock')# file menu

        info = QAction('&Info. and Metadata', self)        
        info.setShortcut('Ctrl+I')
        info.setStatusTip('[Dock] image information dock')
        info.triggered.connect(self.dock.activateINFO)
        dockMenu.addAction(info)

        edit = QAction('&Edit', self)        
        edit.setShortcut('Ctrl+E')
        edit.setStatusTip('[Dock] image editing dock')
        edit.triggered.connect(self.dock.activateEDIT)
        dockMenu.addAction(edit)

        iqa = QAction('&Image Aesthetics', self)        
        iqa.setShortcut('Ctrl+A')
        iqa.setStatusTip('[Dock] image aesthetics dock')
        iqa.triggered.connect(self.dock.activateMIAM)
        dockMenu.addAction(iqa)
    # ------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        """
        Handle application shutdown with proper cleanup.
        
        Saves all ProcessPipe metadata and closes HDR display before exit.
        
        Args:
            event: Qt close event
        """
        if pref.verbose: print(" [CB] >> AppView.closeEvent()>> ... closing")
        self.imageGalleryController.save()
        self.controller.hdrDisplay.close()
# ------------------------------------------------------------------------------------------
# --- class ImageInfoView(QSplitter) -------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageInfoView(QSplitter):
    """
    Image information and metadata display panel.
    
    Provides comprehensive display of image metadata including EXIF data,
    technical parameters, and user-defined tags. Supports interactive editing
    of custom metadata fields for workflow management and image classification.
    
    Layout Structure:
    - Top: Image preview widget
    - Bottom: Scrollable metadata form with:
      * Technical metadata (name, path, size, dynamic range, etc.)
      * EXIF data (exposure, ISO, camera, lens, etc.)
      * User-defined workflow tags (organized by groups)
    
    Attributes:
        - controller (ImageInfoController): Parent controller reference
        - imageWidgetController (ImageWidgetController): Image preview controller
        - layout (QFormLayout): Metadata fields layout
        - imageName, imagePath, imageSize, etc.: Technical metadata widgets
        - userDefinedTags (list[AdvanceCheckBox]): Custom metadata checkboxes
        - scroll (QScrollArea): Scrollable container for metadata fields
    """
    def __init__(self, _controller):
        """
        Initialize image info view with metadata form.
        
        Args:
            _controller (ImageInfoController): Parent controller instance
        """
        if pref.verbose: print(" [VIEW] >> ImageInfoView.__init__(",")")

        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QFormLayout()

        # ---------------------------------------------------
        self.imageName =            AdvanceLineEdit(" name:", " ........ ",             self.layout, callBack=None)
        self.imagePath =            AdvanceLineEdit(" path:", " ........ ",             self.layout, callBack=None)
        self.imageSize =            AdvanceLineEdit(" size (pixel):", ".... x .... ",   self.layout, callBack=None)
        self.imageDynamicRange =    AdvanceLineEdit(" dynamic range (f-stops)", " ........ ", self.layout, callBack=None)
        self.colorSpace =           AdvanceLineEdit(" color space:", " ........ ",      self.layout, callBack=None)
        self.imageType =            AdvanceLineEdit(" type:", " ........ ",             self.layout, callBack=None)                     
        self.imageBPS =             AdvanceLineEdit(" bits per sample:", " ........ ",  self.layout, callBack=None)
        self.imageExpoTime =        AdvanceLineEdit(" exposure time:", " ........ ",    self.layout, callBack=None)
        self.imageFNumber =         AdvanceLineEdit("f-number:", " ........ ",          self.layout, callBack=None)          
        self.imageISO =             AdvanceLineEdit(" ISO:", " ........ ",              self.layout, callBack=None)           
        self.imageCamera =          AdvanceLineEdit(" camera:", " ........ ",           self.layout, callBack=None)      
        self.imageSoftware =        AdvanceLineEdit(" software:", " ........ ",         self.layout, callBack=None)      
        self.imageLens =            AdvanceLineEdit(" lens:", " ........ ",             self.layout, callBack=None)
        self.imageFocalLength =     AdvanceLineEdit(" focal length:", " ........ ",     self.layout, callBack=None)
        # ---------------------------------------------------
        # ---------------------------------------------------
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        self.layout.addRow(line)
        # ---------------------------------------------------
        #  user defined tags
        # --------------------------------------------------
        userDefinedTags = hdrCore.metadata.tags()
        tagRootName = userDefinedTags.getTagsRootName()
        listOfTags = userDefinedTags.tags[tagRootName]
        self.userDefinedTags = []
        for tagGroup in listOfTags:
            groupKey = list(tagGroup.keys())[0]
            tagLeafs = tagGroup[groupKey]
            for tag in tagLeafs.items():
                self.userDefinedTags.append( AdvanceCheckBox(self,groupKey, tag[0], False, self.layout))
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            self.layout.addRow(line)
        # --------------------------------------------------


        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60,40])

    def setProcessPipe(self,processPipe): 
        """
        Update display with new image and populate all metadata fields.
        
        Extracts comprehensive metadata from the ProcessPipe's image including
        EXIF data, technical parameters, and user-defined tags. Updates all
        form fields and checkboxes to reflect current image state.
        
        Args:
            processPipe (ProcessPipe): Processing pipeline containing image with metadata
            
        Returns:
            Result of imageWidgetController.setImage() operation
        """
        image_ = processPipe.getImage()
        # ---------------------------------------------------
        if pref.verbose: print(" [VIEW] >> ImageInfoView.setImage(",image_.name,")")
        if image_.metadata.metadata['filename'] != None: self.imageName.setText(image_.metadata.metadata['filename'])
        else: self.imageName.setText(" ........ ")
        if image_.metadata.metadata['path'] != None: self.imagePath.setText(image_.metadata.metadata['path'])
        else: self.imagePath.setText(" ........ ")
        if image_.metadata.metadata['exif']['Image Width'] != None: self.imageSize.setText(str(image_.metadata.metadata['exif']['Image Width'])+" x "+ str(image_.metadata.metadata['exif']['Image Height']))
        else: self.imageSize.setText(" ........ ")
        if image_.metadata.metadata['exif']['Dynamic Range (stops)'] != None: self.imageDynamicRange.setText(str(image_.metadata.metadata['exif']['Dynamic Range (stops)']))
        else: self.imageDynamicRange.setText(" ........ ")
        if image_.metadata.metadata['exif']['Color Space'] != None: self.colorSpace.setText(image_.metadata.metadata['exif']['Color Space'])
        else: self.imageName.setText(" ........ ")
        if image_.type != None: self.imageType.setText(str(image_.type))       
        else: self.colorSpace.setText(" ........ ")
        if image_.metadata.metadata['exif']['Bits Per Sample'] != None: self.imageBPS.setText(str(image_.metadata.metadata['exif']['Bits Per Sample']))
        else: self.imageBPS.setText(" ........ ")
        if image_.metadata.metadata['exif']['Exposure Time'] != None: self.imageExpoTime.setText(str(image_.metadata.metadata['exif']['Exposure Time'][0])+" / " + str(image_.metadata.metadata['exif']['Exposure Time'][1]))
        else: self.imageExpoTime.setText(" ........ ")
        if image_.metadata.metadata['exif']['F Number'] != None: self.imageFNumber.setText(str(image_.metadata.metadata['exif']['F Number'][0]))    
        else: self.imageFNumber.setText(" ........ ")
        if image_.metadata.metadata['exif']['ISO'] != None: self.imageISO.setText(str(image_.metadata.metadata['exif']['ISO']))       
        else: self.imageISO.setText(" ........ ")
        if image_.metadata.metadata['exif']['Camera'] != None: self.imageCamera.setText(image_.metadata.metadata['exif']['Camera'])      
        else: self.imageCamera.setText(" ........ ")
        if image_.metadata.metadata['exif']['Software'] != None: self.imageSoftware.setText(image_.metadata.metadata['exif']['Software'])      
        else: self.imageSoftware.setText(" ........ ")
        if image_.metadata.metadata['exif']['Lens'] != None: self.imageLens.setText(image_.metadata.metadata['exif']['Lens'])
        else: self.imageLens.setText(" ........ ")
        if image_.metadata.metadata['exif']['Focal Length'] != None: self.imageFocalLength.setText(str(image_.metadata.metadata['exif']['Focal Length'][0]))
        else: self.imageFocalLength.setText(" ........ ")
        # ---------------------------------------------------
        self.controller.callBackActive = False
        # ---------------------------------------------------
        tagRootName = image_.metadata.otherTags.getTagsRootName()
        listOfTags = image_.metadata.metadata[tagRootName]

        for i,tagGroup in enumerate(listOfTags):
            groupKey = list(tagGroup.keys())[0]
            tagLeafs = tagGroup[groupKey]
            for tag in tagLeafs.items():
                # find advanced checkbox
                for acb in self.userDefinedTags:
                    if (acb.rightText ==tag[0] ) and (acb.leftText== groupKey):
                        on_off = image_.metadata.metadata[tagRootName][i][groupKey][tag[0]]
                        on_off = on_off if on_off else False
                        acb.setState(on_off)
                        break  
        # ---------------------------------------------------        
        self.controller.callBackActive = True
        # ---------------------------------------------------
        return self.imageWidgetController.setImage(image_)

    def metadataChange(self,metaGroup,metaTag, on_off): 
        """
        Handle user changes to custom metadata tags.
        
        Args:
            metaGroup (str): Metadata group name
            metaTag (str): Specific tag name
            on_off (bool): New tag state
        """
        if self.controller.callBackActive: self.controller.metadataChange(metaGroup,metaTag, on_off)
# ------------------------------------------------------------------------------------------
# --- class AdvanceLineEdit(object) --------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceLineEdit(object):
    """
    Enhanced line edit widget with label for metadata display.
    
    Provides a labeled text input widget optimized for displaying and
    potentially editing metadata values in forms. Includes automatic
    layout management and optional change callbacks.
    
    Attributes:
        - label (QLabel): Display label for the field
        - lineEdit (QLineEdit): Text input widget
    """
    def __init__(self, labelName, defaultText, layout, callBack=None):
        """
        Initialize labeled line edit widget.
        
        Args:
            labelName (str): Display label text
            defaultText (str): Initial/default text value
            layout (QFormLayout): Parent layout to add widget to
            callBack (function, optional): Text change callback function
        """
        self.label = QLabel(labelName)
        self.lineEdit =QLineEdit(defaultText)
        if callBack: self.lineEdit.textChanged.connect(callBack)
        layout.addRow(self.label,self.lineEdit)

    def setText(self, txt): 
        """
        Update the displayed text value.
        
        Args:
            txt (str): New text to display
        """
        self.lineEdit.setText(txt)
# ------------------------------------------------------------------------------------------
# --- class AdvanceCheckBox(object) --------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceCheckBox(object):
    """
    Enhanced checkbox widget with label for metadata editing.
    
    Provides a labeled checkbox widget optimized for editing user-defined
    metadata tags. Supports callback functionality for metadata updates
    and automatic layout management.
    
    Attributes:
        - parent (ImageInfoView): Parent view for callback delegation
        - leftText (str): Left label text (group name)
        - rightText (str): Right label text (tag name)
        - label (QLabel): Display label widget
        - checkbox (QCheckBox): Checkbox control
    """
    def __init__(self, parent, leftText, rightText, defaultValue, layout):
        """
        Initialize labeled checkbox widget.
        
        Args:
            parent (ImageInfoView): Parent view for callbacks
            leftText (str): Left label text (metadata group)
            rightText (str): Right label text (metadata tag)
            defaultValue (bool): Initial checkbox state
            layout (QFormLayout): Parent layout to add widget to
        """
        self.parent = parent

        self.leftText = leftText
        self.rightText = rightText

        self.label = QLabel(leftText)
        self.checkbox =QCheckBox(rightText)
        self.checkbox.toggled.connect(self.toggled)
        layout.addRow(self.label,self.checkbox)

    def setState(self, on_off): 
        """
        Set checkbox checked state.
        
        Args:
            on_off (bool): New checked state
        """
        self.checkbox.setChecked(on_off)

    def toggled(self): 
        """
        Handle checkbox state changes and notify parent.
        """
        self.parent.metadataChange(self.leftText, self.rightText, self.checkbox.isChecked())
# ------------------------------------------------------------------------------------------
# --- class EditImageView(QSplitter) -------------------------------------------------------
# ------------------------------------------------------------------------------------------
class EditImageView(QSplitter):
    """
    Comprehensive HDR image editing interface with all editing controls.
    
    Provides the complete HDR editing interface with real-time preview and
    extensive parameter controls. Includes exposure, contrast, tone curves,
    color editing, geometry transformations, and HDR preview capabilities.
    
    Layout Structure:
    - Top: Image preview showing real-time edits
    - Bottom: Scrollable controls panel with:
      * Basic adjustments (exposure, contrast, saturation)
      * Advanced tone curve with B-spline control points
      * Lightness masking for selective editing
      * Five independent color editors in LCH space
      * Automatic color selection tools
      * Geometry transformations (rotation, cropping)
      * HDR preview controls
    
    Attributes:
        - controller (EditImageController): Parent controller reference
        - imageWidgetController (ImageWidgetController): Image preview controller
        - exposure, contrast, saturation (AdvanceSliderController): Basic adjustment controls
        - tonecurve (ToneCurveController): B-spline tone curve editor
        - lightnessmask (LightnessMaskController): Tone range masking
        - colorEditor0-4 (LchColorSelectorController): Independent color editors
        - colorEditorsAuto (ColorEditorsAutoController): Automatic color selection
        - geometry (GeometryController): Rotation and cropping controls
        - hdrPreview (HDRviewerView): HDR display preview interface
        - scroll (QScrollArea): Scrollable container for all controls
    """

    def __init__(self, _controller, build=False):
        """
        Initialize comprehensive HDR editing interface.
        
        Creates all editing controls and sets up the scrollable interface
        with proper callback connections for real-time parameter updates.
        
        Args:
            _controller (EditImageController): Parent controller instance
            build (bool): Whether to restore previous state during construction
        """
        if pref.verbose: print(" [VIEW] >> EditImageView.__init__(",")")
        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QVBoxLayout()

        # exposure ----------------------
        self.exposure = controller.AdvanceSliderController(self, "exposure",0,(-10,+10),0.25)
        # call back functions
        self.exposure.callBackAutoPush = self.autoExposure
        self.exposure.callBackValueChange = self.changeExposure
        self.layout.addWidget(self.exposure.view)

        # contrast ----------------------
        self.contrast = controller.AdvanceSliderController(self, "contrast",0,(-100,+100),1)
        # call back functions
        self.contrast.callBackAutoPush = self.autoContrast
        self.contrast.callBackValueChange = self.changeContrast
        self.layout.addWidget(self.contrast.view)

        # tonecurve ----------------------
        self.tonecurve = controller.ToneCurveController(self)        
        self.layout.addWidget(self.tonecurve.view)                   

        # mask ----------------------
        self.lightnessmask = controller.LightnessMaskController(self)        
        self.layout.addWidget(self.lightnessmask.view)                   
 
        # saturation ----------------------
        self.saturation = controller.AdvanceSliderController(self, "saturation",0,(-100,+100),1)
        # call back functions
        self.saturation.callBackAutoPush = self.autoSaturation
        self.saturation.callBackValueChange = self.changeSaturation
        self.layout.addWidget(self.saturation.view)

        # color0 ----------------------
        self.colorEditor0 = controller.LchColorSelectorController(self, idName = "colorEditor0")
        self.layout.addWidget(self.colorEditor0.view)

        # color1 ----------------------
        self.colorEditor1 = controller.LchColorSelectorController(self, idName = "colorEditor1")
        self.layout.addWidget(self.colorEditor1.view)

        # color2 ----------------------
        self.colorEditor2 = controller.LchColorSelectorController(self, idName = "colorEditor2")
        self.layout.addWidget(self.colorEditor2.view)

        # color3 ----------------------
        self.colorEditor3 = controller.LchColorSelectorController(self, idName = "colorEditor3")
        self.layout.addWidget(self.colorEditor3.view)

        # color1 ----------------------
        self.colorEditor4 = controller.LchColorSelectorController(self, idName = "colorEditor4")
        self.layout.addWidget(self.colorEditor4.view)

        # auto color selection ----------------------
        # -----
        self.colorEditorsAuto = controller.ColorEditorsAutoController(self,
                                                                      [self.colorEditor0,
                                                                       self.colorEditor1,
                                                                       self.colorEditor2,
                                                                       self.colorEditor3,
                                                                       self.colorEditor4],
                                                                       "saturation")
        self.layout.addWidget(self.colorEditorsAuto.view)
        # -----

        # geometry ----------------------
        self.geometry = controller.GeometryController(self)
        self.layout.addWidget(self.geometry.view)

        # hdr preview ----------------------
        self.hdrPreview = HDRviewerView(self.controller.controllerHDR, build)
        self.controller.controllerHDR.setView(self.hdrPreview)
        self.layout.addWidget(self.hdrPreview)

        # scroll ------------------------------------------------
        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        # adding widgets to self (QSplitter)
        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60,40])

    def setImage(self,image):
        """
        Update the image preview display.
        
        Args:
            image (hdrCore.image.Image): Image to display in preview
            
        Returns:
            Result of imageWidgetController.setImage() operation
        """
        if pref.verbose: print(" [VIEW] >> EditImageView.setImage(",image.name,")")
        return self.imageWidgetController.setImage(image)

    def autoExposure(self):
        """
        Trigger automatic exposure calculation.
        
        Delegates to controller for automatic exposure value computation
        based on image histogram analysis.
        """
        if pref.verbose: print(" [CB] >> EditImageView.autoExposure(",")")

        self.controller.autoExposure()
        pass

    def changeExposure(self, value):
        """
        Handle manual exposure adjustment changes.
        
        Args:
            value (float): New exposure value in EV stops
        """
        if pref.verbose: print(" [CB] >> EditImageView.changeExposure(",")")

        self.controller.changeExposure(value)
        pass

    def autoContrast(self):
        """
        Trigger automatic contrast calculation.
        
        Placeholder for automatic contrast adjustment functionality.
        """
        if pref.verbose: print(" [CB] >> EditImageView.autoContrast(",")")
        pass

    def changeContrast(self, value):
        """
        Handle manual contrast adjustment changes.
        
        Args:
            value (float): New contrast value
        """
        if pref.verbose: print(" [CB] >> EditImageView.changeContrast(",")")

        self.controller.changeContrast(value)
        pass

    def autoSaturation(self):
        """
        Trigger automatic saturation calculation.
        
        Placeholder for automatic saturation adjustment functionality.
        """
        print(" [CB] >> EditImageView.autoSaturation(",")")
        pass

    def changeSaturation(self,value):   ### TO DO
        """
        Handle manual saturation adjustment changes.
        
        Args:
            value (float): New saturation value
        """
        if pref.verbose:  print(" [CB] >> EditImageView.changeSaturation(",")")
        self.controller.changeSaturation(value)

    def plotToneCurve(self): 
        """
        Update tone curve visualization in the curve editor.
        """
        self.tonecurve.plotCurve()
 
    def setProcessPipe(self, processPipe):
        """
        Initialize interface with ProcessPipe parameters.
        
        Recovers all editing parameters from the ProcessPipe and restores
        the interface state to match the current image's editing configuration.
        This method is called when switching between images to maintain
        consistent editing state.
        
        Args:
            processPipe (ProcessPipe): Processing pipeline with current parameters
        """
        if pref.verbose:  print(" [VIEW] >> EditImageView.setProcessPipe(",")")

        # exposure
        # recover value in pipe and restore it
        id = processPipe.getProcessNodeByName("exposure")
        value = processPipe.getParameters(id)
        self.exposure.setValue(value['EV'], callBackActive = False)

        # contrast
        # recover value in pipe and restore it
        id = processPipe.getProcessNodeByName("contrast")
        value = processPipe.getParameters(id)
        self.contrast.setValue(value['contrast'], callBackActive = False)

        # tonecurve
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("tonecurve")
        value = processPipe.getParameters(id)
        self.tonecurve.setValues(value,callBackActive = False)

        # lightnessmask
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("lightnessmask")
        value = processPipe.getParameters(id)
        self.lightnessmask.setValues(value,callBackActive = False)

        # saturation
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("saturation")
        value = processPipe.getParameters(id)
        self.saturation.setValue(value['saturation'], callBackActive = False)

        # colorEditor0
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor0")
        values = processPipe.getParameters(id)
        self.colorEditor0.setValues(values, callBackActive = False)

        # colorEditor1
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor1")
        values = processPipe.getParameters(id)
        self.colorEditor1.setValues(values, callBackActive = False)

        # colorEditor2
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor2")
        values = processPipe.getParameters(id)
        self.colorEditor2.setValues(values, callBackActive = False)

        # colorEditor3
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor3")
        values = processPipe.getParameters(id)
        self.colorEditor3.setValues(values, callBackActive = False)
        
        # colorEditor4
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor4")
        values = processPipe.getParameters(id)
        self.colorEditor4.setValues(values, callBackActive = False)
        
        # geometry
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("geometry")
        values = processPipe.getParameters(id)
        self.geometry.setValues(values, callBackActive = False)
# ------------------------------------------------------------------------------------------
# --- class MultiDockView(QDockWidget) -----------------------------------------------------
# ------------------------------------------------------------------------------------------
class MultiDockView(QDockWidget):
    """
    Multi-panel dockable widget for uHDR right-side interface.
    
    Provides a switchable dock widget that can display different interface panels
    for HDR image editing workflow. Manages three distinct child controllers
    that handle different aspects of image processing: editing controls, metadata
    information, and aesthetics analysis.
    
    Panel Types:
    - Panel 0: Edit Image Controls (EditImageController)
      * HDR parameter adjustments (exposure, contrast, saturation)
      * Tone curve editing with B-spline controls
      * Color space editing in LCH coordinates
      * Geometry transformations and HDR preview
      
    - Panel 1: Image Information & Metadata (ImageInfoController)
      * EXIF data display and technical parameters
      * User-defined workflow tags and classification
      * File information and processing metadata
      
    - Panel 2: Image Aesthetics Analysis (ImageAestheticsController)
      * Color palette extraction using K-means clustering
      * Dominant color visualization and analysis
      * Aesthetic composition metrics
    
    Architecture:
    The MultiDockView implements a container pattern where child controllers
    manage their own views and business logic. The dock handles view switching,
    memory management, and state synchronization between panels.
    
    Docking Behavior:
    - Allowed areas: Left or Right dock areas only
    - Default position: Right side of main window
    - Resizable and detachable following Qt dock widget standards
    - Panel switching preserves current ProcessPipe state
    
    State Management:
    When switching panels, the dock:
    1. Destroys the current view to free memory
    2. Rebuilds the target view with current ProcessPipe
    3. Updates the dock widget content
    4. Maintains consistent state across panels
    
    Memory Optimization:
    Views are dynamically created/destroyed during switching to minimize
    memory usage, especially important for the complex editing interface
    with matplotlib components and HDR preview capabilities.
    
    Attributes:
        - controller (MultiDockController): Parent controller for dock management
        - childControllers (list): List of three child controllers:
            [0] EditImageController - editing interface
            [1] ImageInfoController - metadata interface  
            [2] ImageAestheticsController - aesthetics interface
        - childController: Currently active child controller
        - active (int): Index of currently displayed panel (0, 1, or 2)
    
    Usage Example:
        # Create multi-dock with HDR support
        dock = MultiDockView(controller, HDRcontroller)
        
        # Switch to metadata panel
        dock.switch(1)
        
        # Update with new image
        dock.setProcessPipe(processPipe)
    
    Integration:
    The MultiDockView integrates with the main AppView through menu
    shortcuts (Ctrl+E, Ctrl+I, Ctrl+A) that trigger panel switching
    for efficient workflow navigation.
    """
    def __init__(self, _controller, HDRcontroller=None):
        """
        Initialize multi-panel dock widget with child controllers.
        
        Creates the dock widget with three child controllers for different
        interface panels. Sets up docking constraints, initializes the
        editing panel as default, and establishes parent-child relationships.
        
        Args:
            _controller (MultiDockController): Parent controller managing dock behavior
            HDRcontroller (HDRController, optional): HDR display controller for
                real-time HDR preview functionality in editing panel
                
        Child Controllers Created:
            - EditImageController: Complete HDR editing interface with tone curves,
              color editors, geometry controls, and HDR preview
            - ImageInfoController: Metadata display and user tag management
            - ImageAestheticsController: Color palette and aesthetic analysis
        """
        if pref.verbose:  print(" [VIEW] >> MultiDockView.__init__(",")")

        super().__init__("Image Edit/Info")
        self.controller = _controller

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
      
        self.childControllers = [
            controller.EditImageController(self, HDRcontroller), 
            controller.ImageInfoController(self), 
            controller.ImageAestheticsController(self)]
            #controller.ImageQualityController(self, HDRcontroller)]
        self.childController = self.childControllers[0]
        self.active = 0
        self.setWidget(self.childController.view)
        self.repaint()
    # ------------------------------------------------------------------------------------------
    def switch(self,nb):
        """
        Switch to different dock panel and rebuild interface.
        
        Changes the active dock panel by destroying the current view,
        selecting the target child controller, rebuilding the view with
        current ProcessPipe state, and updating the dock widget content.
        
        Panel Selection:
        - nb = 0: Edit Image Controls
          * HDR parameter adjustments (exposure, contrast, saturation)
          * Tone curve editing and lightness masking
          * Color space editing in LCH coordinates
          * Geometry transformations and HDR preview
          
        - nb = 1: Image Information & Metadata  
          * EXIF data display and technical parameters
          * User-defined workflow tags and classification
          * File information and processing metadata
          
        - nb = 2: Image Aesthetics Analysis
          * Color palette extraction using K-means clustering
          * Dominant color visualization and analysis
          * Aesthetic composition analysis
        
        Memory Management:
        The method performs proper cleanup by calling deleteLater() on the
        current view before creating the new one, preventing memory leaks
        from complex widgets like matplotlib figures and HDR previews.
        
        Args:
            nb (int): Target panel index (0, 1, or 2). Values outside range
                     are wrapped using modulo operation for safety.
                     
        State Preservation:
        Current ProcessPipe is retrieved and passed to the new panel to
        maintain consistent state across interface switches.
        """
        if pref.verbose:  print(" [VIEW] >> MultiDockView.switch(",nb,")")

        if nb != self.active:
     
            self.active = (nb)%len(self.childControllers)
            self.childController.view.deleteLater()
            self.childController = self.childControllers[self.active]
            # rebuild view 
            processPipe = self.controller.parent.imageGalleryController.getSelectedProcessPipe()
            self.childController.buildView(processPipe)
            self.setWidget(self.childController.view)
            self.repaint()
    # ------------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe):
        """
        Update current panel with new ProcessPipe data.
        
        Delegates ProcessPipe updates to the currently active child controller,
        allowing each panel to refresh its content and interface state based
        on the new image and processing parameters.
        
        Panel-Specific Behavior:
        - Edit Panel: Restores all editing parameters (exposure, contrast, 
          tone curves, color editors, geometry) from ProcessPipe nodes
        - Info Panel: Updates metadata display, EXIF data, and user tags
        - Aesthetics Panel: Regenerates color palette and analysis visualization
        
        Args:
            processPipe (ProcessPipe): Processing pipeline containing image
                and all associated editing parameters and metadata
                
        Returns:
            Result of active child controller's setProcessPipe() method,
            typically the updated image or processing result
            
        Note:
            This method provides a unified interface for ProcessPipe updates
            regardless of which panel is currently active, simplifying the
            dock management logic in the parent controller.
        """
        if pref.verbose:  print(" [VIEW] >> MultiDockView.setProcessPipe(",processPipe.getImage().name,")")
        return self.childController.setProcessPipe(processPipe)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# --- class AdvanceSliderView(QFrame) ------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceSliderView(QFrame):
    """
    Enhanced slider widget with auto/reset functionality and numerical input.
    
    Provides a comprehensive parameter adjustment interface combining a horizontal
    slider with automatic calculation, manual reset, and direct numerical input
    capabilities. Used extensively throughout uHDR for real-time parameter
    adjustments in exposure, contrast, saturation, and other image processing
    controls.
    
    Widget Components:
    - Descriptive label: Parameter name display
    - Auto button: Triggers automatic parameter calculation
    - Value editor: Direct numerical input with validation
    - Reset button: Restores default parameter value
    - Horizontal slider: Continuous parameter adjustment
    
    Value Handling:
    The widget internally scales slider positions based on step size for
    precise control. Slider range is calculated as (range[0]/step, range[1]/step)
    to provide appropriate granularity for different parameter types.
    
    Layout Structure:
    +------------------------------------------+
    | [Label] [Auto] [Value Input] [Reset]    |  <- First row (horizontal)
    +------------------------------------------+
    | [========== Slider =================]   |  <- Second row
    +------------------------------------------+
    
    Callback Integration:
    All user interactions automatically trigger controller callbacks:
    - Slider movement: controller.sliderChange()
    - Reset button: controller.reset()
    - Auto button: controller.auto()
    
    Value input changes are handled through slider synchronization.
    
    Attributes:
        - controller: Parent controller for callback delegation
        - firstrow (QFrame): Container for top row controls
        - label (QLabel): Parameter name display
        - auto (QPushButton): Automatic parameter calculation trigger
        - editValue (QLineEdit): Direct numerical input field with validation
        - reset (QPushButton): Default value restoration button
        - slider (QSlider): Main parameter adjustment control
        - vbox (QVBoxLayout): Vertical layout manager
        - hbox (QHBoxLayout): Horizontal layout for first row
    
    Example Usage:
        # Exposure control with ±10 EV range and 0.25 step
        exposure_slider = AdvanceSliderView(controller, "exposure", 0, (-10, 10), 0.25)
        
        # Contrast control with ±100 range and 1.0 step  
        contrast_slider = AdvanceSliderView(controller, "contrast", 0, (-100, 100), 1.0)
    
    Technical Notes:
        - QDoubleValidator ensures only valid numerical input
        - Step size determines slider precision and scaling
        - Range values should be compatible with step size
        - Controller callbacks handle actual parameter application
        - Widget maintains internal consistency between slider and text input
    """
    def __init__(self, controller, name,defaultValue, range, step):
        """
        Initialize enhanced slider widget with auto/reset functionality.
        
        Creates a complete parameter adjustment interface with slider, buttons,
        and text input. Automatically configures scaling, validation, and
        callback connections for seamless integration.
        
        Args:
            controller: Parent controller object implementing callback methods:
                - sliderChange(): Handle slider value changes
                - reset(): Restore default parameter value  
                - auto(): Calculate automatic parameter value
            name (str): Human-readable parameter name for label display
            defaultValue (float): Initial parameter value and reset target
            range (tuple): Parameter range as (min_value, max_value)
            step (float): Parameter adjustment granularity and slider scaling factor
                
        Example:
            # Create exposure slider: -10 to +10 EV, 0.25 EV steps
            slider = AdvanceSliderView(controller, "exposure", 0.0, (-10.0, 10.0), 0.25)
            
            # Create contrast slider: -100 to +100, 1.0 steps  
            slider = AdvanceSliderView(controller, "contrast", 0.0, (-100.0, 100.0), 1.0)
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = controller
        self.firstrow = QFrame()

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()
        
        self.firstrow.setLayout(self.hbox)

        self.label= QLabel(name)
        self.auto = QPushButton("auto")
        self.editValue = QLineEdit()
        self.editValue.setValidator(QDoubleValidator())
        self.editValue.setText(str(defaultValue))
        self.reset = QPushButton("reset")

        self.hbox.addWidget(self.label)
        self.hbox.addWidget(self.auto)
        self.hbox.addWidget(self.editValue)
        self.hbox.addWidget(self.reset)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(range[0]/step),int(range[1]/step))
        self.slider.setValue(int(defaultValue/step))
        self.slider.setSingleStep(1) 

        self.vbox.addWidget(self.firstrow)
        self.vbox.addWidget(self.slider)

        self.setLayout(self.vbox)

        # callBackFunctions slider/reset/auto
        self.slider.valueChanged.connect(self.controller.sliderChange)
        self.reset.clicked.connect(self.controller.reset)
        self.auto.clicked.connect(self.controller.auto)
# ------------------------------------------------------------------------------------------
# --- class ToneCurveView(QFrame) ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ToneCurveView(QFrame):
    """
    B-spline tone curve editing interface with matplotlib visualization.
    
    Provides comprehensive tone curve editing capabilities using B-spline interpolation
    with individual control over five tonal ranges: shadows, blacks, mediums, whites,
    and highlights. Features real-time curve visualization and semi-automatic curve
    generation for enhanced workflow efficiency.
    
    Tone Curve Controls:
    - Shadows (0-20%): Darkest image regions control
    - Blacks (20-40%): Dark tone adjustments  
    - Mediums (40-60%): Mid-tone contrast and brightness
    - Whites (60-80%): Bright tone refinement
    - Highlights (80-100%): Brightest regions control
    
    Interface Features:
    - Real-time B-spline curve visualization using matplotlib
    - Individual sliders for each tonal range (0-100 scale)
    - Reset buttons for selective parameter restoration
    - Semi-automatic curve generation based on image analysis
    - Live curve updates during parameter adjustment
    
    B-spline Implementation:
    The tone curve uses B-spline interpolation with fixed control points
    at the five tonal boundaries. User adjustments modify the Y-coordinates
    of these control points while maintaining smooth curve transitions.
    
    Mathematical Foundation:
    - Input domain: [0, 100] (percentage lightness)
    - Output range: [0, 100] (adjusted lightness)
    - Control points: (0,0), (20,shadows), (40,blacks), (60,mediums), (80,whites), (100,highlights)
    - Interpolation: Cubic B-spline with C² continuity
    
    Layout Structure:
    +------------------------------------------+
    | [Matplotlib Curve Visualization]        |
    +------------------------------------------+
    | [Auto Curve Generation Button]          |
    +------------------------------------------+
    | highlights: [slider] [value] [reset]    |
    | whites:     [slider] [value] [reset]    |
    | mediums:    [slider] [value] [reset]    |
    | blacks:     [slider] [value] [reset]    |
    | shadows:    [slider] [value] [reset]    |
    +------------------------------------------+
    
    Attributes:
        - controller (ToneCurveController): Parent controller for curve calculations
        - curve (FigureWidget): Matplotlib figure widget for curve display
        - autoCurve (QPushButton): Semi-automatic curve generation button
        - sliderShadows, sliderBlacks, sliderMediums, sliderWhites, sliderHighlights: 
            Individual tone range control sliders (0-100 range)
        - editShadows, editBlacks, editMediums, editWhites, editHighlights:
            Text input fields showing current slider values
        - resetShadows, resetBlacks, resetMediums, resetWhites, resetHighlights:
            Reset buttons for individual tone ranges
        - labelShadows, labelBlacks, labelMediums, labelWhites, labelHighlights:
            Descriptive labels for each tone range
    
    Usage Example:
        The tone curve allows precise control over image tonality:
        - Lift shadows (increase slider) to reveal dark detail
        - Lower highlights (decrease slider) to recover bright detail  
        - Adjust mediums for overall contrast and brightness
        - Use auto curve for intelligent initial settings
        
    Technical Notes:
        - All slider values represent percentage adjustments (0-100)
        - Curve updates trigger real-time image processing
        - B-spline ensures smooth transitions between control points
        - Auto curve analyzes image histogram for optimal settings
    """

    def __init__(self, controller):
        """
        Initialize B-spline tone curve editing interface.
        
        Creates the complete tone curve editing interface with matplotlib
        visualization and individual control sliders for five tonal ranges.
        Sets up all UI components, connects callbacks, and initializes
        default parameter values.
        
        Args:
            controller (ToneCurveController): Parent controller instance
                that handles curve calculations and image processing
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)

        self.controller = controller

        self.vbox = QVBoxLayout()

        # figure
        self.curve = FigureWidget(self) 
        self.curve.setMinimumSize(200, 600)

        self.curve.plot([0.0,100],[0.0,100.0],'r--')

        #containers
        
        # zj add for semi-auto curve begin
        self.containerAuto = QFrame() 
        self.hboxAuto = QHBoxLayout() 
        self.containerAuto.setLayout(self.hboxAuto)
        # zj add for semi-auto curve end                                
        self.containerShadows = QFrame()
        self.hboxShadows = QHBoxLayout()
        self.containerShadows.setLayout(self.hboxShadows)
        
        self.containerBlacks = QFrame()
        self.hboxBlacks = QHBoxLayout()
        self.containerBlacks.setLayout(self.hboxBlacks)

        self.containerMediums = QFrame()
        self.hboxMediums = QHBoxLayout()
        self.containerMediums.setLayout(self.hboxMediums)

        self.containerWhites = QFrame()
        self.hboxWhites = QHBoxLayout()
        self.containerWhites.setLayout(self.hboxWhites)

        self.containerHighlights = QFrame()
        self.hboxHighlights = QHBoxLayout()
        self.containerHighlights.setLayout(self.hboxHighlights)

        self.vbox.addWidget(self.curve)
        self.vbox.addWidget(self.containerAuto) #  zj add for semi-auto curve                                                                       
        self.vbox.addWidget(self.containerHighlights)
        self.vbox.addWidget(self.containerWhites)
        self.vbox.addWidget(self.containerMediums)
        self.vbox.addWidget(self.containerBlacks)
        self.vbox.addWidget(self.containerShadows)

        # zj add for semi-auto curve begin
        # autoCurve button
        self.autoCurve = QPushButton("auto")
        self.hboxAuto.addWidget(self.autoCurve)
        self.hboxAuto.setAlignment(Qt.AlignCenter)
        # zj add for semi-auto curve end
        # shadows
        self.labelShadows = QLabel("shadows")
        self.sliderShadows = QSlider(Qt.Horizontal)
        self.sliderShadows.setRange(0,100)
        self.sliderShadows.setValue(int(self.controller.model.default['shadows'][1]))
        self.editShadows = QLineEdit()
        self.editShadows.setText(str(self.controller.model.default['shadows'][1]))
        self.resetShadows = QPushButton("reset")
        self.hboxShadows.addWidget(self.labelShadows)
        self.hboxShadows.addWidget(self.sliderShadows)
        self.hboxShadows.addWidget(self.editShadows)
        self.hboxShadows.addWidget(self.resetShadows)

        # blacks
        self.labelBlacks = QLabel("  blacks  ")
        self.sliderBlacks = QSlider(Qt.Horizontal)
        self.sliderBlacks.setRange(0,100)
        self.sliderBlacks.setValue(int(self.controller.model.default['blacks'][1]))
        self.editBlacks = QLineEdit()
        self.editBlacks.setText(str(self.controller.model.default['blacks'][1]))
        self.resetBlacks = QPushButton("reset")
        self.hboxBlacks.addWidget(self.labelBlacks)
        self.hboxBlacks.addWidget(self.sliderBlacks)
        self.hboxBlacks.addWidget(self.editBlacks)
        self.hboxBlacks.addWidget(self.resetBlacks)

        # mediums
        self.labelMediums = QLabel("mediums")
        self.sliderMediums = QSlider(Qt.Horizontal)
        self.sliderMediums.setRange(0,100)
        self.sliderMediums.setValue(int(self.controller.model.default['mediums'][1]))
        self.editMediums = QLineEdit()
        self.editMediums.setText(str(self.controller.model.default['mediums'][1]))
        self.resetMediums = QPushButton("reset")
        self.hboxMediums.addWidget(self.labelMediums)
        self.hboxMediums.addWidget(self.sliderMediums)
        self.hboxMediums.addWidget(self.editMediums)
        self.hboxMediums.addWidget(self.resetMediums)

        # whites
        self.labelWhites = QLabel("  whites  ")
        self.sliderWhites = QSlider(Qt.Horizontal)
        self.sliderWhites.setRange(0,100)
        self.sliderWhites.setValue(int(self.controller.model.default['whites'][1]))
        self.editWhites = QLineEdit()
        self.editWhites.setText(str(self.controller.model.default['whites'][1]))
        self.resetWhites = QPushButton("reset")
        self.hboxWhites.addWidget(self.labelWhites)
        self.hboxWhites.addWidget(self.sliderWhites)
        self.hboxWhites.addWidget(self.editWhites)
        self.hboxWhites.addWidget(self.resetWhites)

        # highlights
        self.labelHighlights = QLabel("highlights")
        self.sliderHighlights = QSlider(Qt.Horizontal)
        self.sliderHighlights.setRange(0,100)
        self.sliderHighlights.setValue(int(self.controller.model.default['highlights'][1]))
        self.editHighlights = QLineEdit()
        self.editHighlights.setText(str(self.controller.model.default['highlights'][1]))
        self.resetHighlights = QPushButton("reset")
        self.hboxHighlights.addWidget(self.labelHighlights)
        self.hboxHighlights.addWidget(self.sliderHighlights)
        self.hboxHighlights.addWidget(self.editHighlights)
        self.hboxHighlights.addWidget(self.resetHighlights)

        self.setLayout(self.vbox)

        # callBackFunctions slider/reset
        self.sliderShadows.valueChanged.connect(self.sliderShadowsChange)
        self.sliderBlacks.valueChanged.connect(self.sliderBlacksChange)
        self.sliderMediums.valueChanged.connect(self.sliderMediumsChange)
        self.sliderWhites.valueChanged.connect(self.sliderWhitesChange)
        self.sliderHighlights.valueChanged.connect(self.sliderHighlightsChange)

        self.resetShadows.clicked.connect(self.resetShadowsCB)
        self.resetBlacks.clicked.connect(self.resetBlacksCB)
        self.resetMediums.clicked.connect(self.resetMediumsCB)
        self.resetWhites.clicked.connect(self.resetWhitesCB)
        self.resetHighlights.clicked.connect(self.resetHighlightsCB)
        self.autoCurve.clicked.connect(self.controller.autoCurve)  #  zj add for semi-auto curve 
                                                                                         

    def sliderShadowsChange(self):
        """
        Handle shadows slider value changes and update tone curve.
        
        Called when the shadows slider (0-20% tonal range) is moved.
        Updates the curve visualization and triggers image processing
        if callbacks are active.
        """
        if self.controller.callBackActive:
            value = self.sliderShadows.value()
            self.controller.sliderChange("shadows", value)

    def sliderBlacksChange(self):
        """
        Handle blacks slider value changes and update tone curve.
        
        Called when the blacks slider (20-40% tonal range) is moved.
        Updates the curve visualization and triggers image processing
        if callbacks are active.
        """
        if self.controller.callBackActive:
            value = self.sliderBlacks.value()
            self.controller.sliderChange("blacks", value)

    def sliderMediumsChange(self):
        """
        Handle mediums slider value changes and update tone curve.
        
        Called when the mediums slider (40-60% tonal range) is moved.
        This controls mid-tone contrast and overall image brightness.
        Updates the curve visualization and triggers image processing
        if callbacks are active.
        """
        if self.controller.callBackActive:
            value = self.sliderMediums.value()
            self.controller.sliderChange("mediums", value)

    def sliderWhitesChange(self):
        """
        Handle whites slider value changes and update tone curve.
        
        Called when the whites slider (60-80% tonal range) is moved.
        Used for bright tone refinement and highlight detail control.
        Updates the curve visualization and triggers image processing
        if callbacks are active.
        """
        if self.controller.callBackActive:
            value = self.sliderWhites.value()
            self.controller.sliderChange("whites", value)

    def sliderHighlightsChange(self):
        """
        Handle highlights slider value changes and update tone curve.
        
        Called when the highlights slider (80-100% tonal range) is moved.
        Controls the brightest image regions and highlight detail recovery.
        Updates the curve visualization and triggers image processing
        if callbacks are active.
        """
        if self.controller.callBackActive:
            value = self.sliderHighlights.value()
            self.controller.sliderChange("highlights", value)

    def resetShadowsCB(self):
        """Reset shadows slider to default value and update curve."""
        if self.controller.callBackActive: self.controller.reset("shadows")

    def resetBlacksCB(self):
        """Reset blacks slider to default value and update curve."""
        if self.controller.callBackActive: self.controller.reset("blacks")
    
    def resetMediumsCB(self):
        """Reset mediums slider to default value and update curve."""
        if self.controller.callBackActive: self.controller.reset("mediums")

    def resetWhitesCB(self):
        """Reset whites slider to default value and update curve."""
        if self.controller.callBackActive: self.controller.reset("whites")

    def resetHighlightsCB(self):
        """Reset highlights slider to default value and update curve."""
        if self.controller.callBackActive: self.controller.reset("highlights")
# ------------------------------------------------------------------------------------------
# --- class LightnessMaskView(QGroupBox) ---------------------------------------------------
# ------------------------------------------------------------------------------------------
class LightnessMaskView(QGroupBox):
    """
    Lightness-based masking control interface for selective tone editing.
    
    Provides checkbox controls for enabling/disabling tone range masks that allow
    selective application of editing adjustments to specific lightness regions.
    Works in conjunction with tone curve editing to provide precise control over
    which tonal ranges are affected by parameter changes.
    
    Tone Range Masking:
    The interface allows independent masking of five distinct lightness ranges
    corresponding to the tone curve control points:
    
    - Shadows (0-20%): Darkest image regions
      * Deep shadows, true blacks, dark details
      * Useful for shadow lifting without affecting other tones
      
    - Blacks (20-40%): Lower mid-tone range
      * Dark greys, low-key lighting areas
      * Controls separation between shadows and mid-tones
      
    - Mediums (40-60%): Mid-tone range
      * Standard exposure areas, skin tones, neutral greys
      * Primary brightness and contrast control zone
      
    - Whites (60-80%): Upper mid-tone range  
      * Bright areas, light greys, highlight transitions
      * Controls separation between mid-tones and highlights
      
    - Highlights (80-100%): Brightest image regions
      * Specular highlights, light sources, bright reflections
      * Useful for highlight recovery and bright detail control
    
    Masking Functionality:
    When checkboxes are enabled, subsequent editing operations (exposure,
    contrast, color adjustments) are selectively applied only to pixels
    within the specified lightness ranges. This allows for:
    
    - Targeted shadow lifting without brightening highlights
    - Selective highlight recovery without darkening shadows
    - Mid-tone contrast adjustments without clipping extremes
    - Color corrections limited to specific tonal ranges
    - Fine-tuned local adjustments for advanced workflow
    
    Layout Structure:
    +---------------------------------------------------------------+
    | mask lightness                                               |
    +---------------------------------------------------------------+
    | [✓] shadows [✓] blacks [✓] mediums [✓] whites [✓] highlights|
    +---------------------------------------------------------------+
    
    Integration:
    The mask settings work seamlessly with other editing controls in the
    EditImageView, providing a non-destructive workflow where masks can
    be enabled, adjusted, and disabled without losing editing progress.
    
    Attributes:
        - controller (LightnessMaskController): Parent controller for mask logic
        - checkboxShadows (QCheckBox): Shadows range mask toggle
        - checkboxBlacks (QCheckBox): Blacks range mask toggle  
        - checkboxMediums (QCheckBox): Mediums range mask toggle
        - checkboxWhites (QCheckBox): Whites range mask toggle
        - checkboxHighlights (QCheckBox): Highlights range mask toggle
        - hbox (QHBoxLayout): Horizontal layout for checkbox arrangement

    Usage Example:
        # Enable shadow and highlight masking for targeted adjustment
        mask_view.checkboxShadows.setChecked(True)
        mask_view.checkboxHighlights.setChecked(True)
        
        # Apply selective exposure adjustment
        # (affects only shadows and highlights)
        
    Technical Notes:
        - Mask ranges correspond to tone curve control points
        - Multiple masks can be active simultaneously
        - Masks affect all subsequent editing operations
        - Checkbox state is preserved in ProcessPipe parameters
        - Real-time mask preview available through color editors
    """
    def __init__(self, _controller):
        """
        Initialize lightness masking interface with five tone range checkboxes.
        
        Creates a horizontal arrangement of checkboxes for controlling tone range
        masks. Sets up callback connections for real-time mask updates and
        initializes all masks as disabled by default.
        
        Args:
            _controller (LightnessMaskController): Parent controller implementing
                maskChange() method for handling mask state updates
                
        Interface Layout:
            All checkboxes are arranged horizontally in reading order:
            [Shadows] [Blacks] [Mediums] [Whites] [Highlights]
            
        Default State:
            All masks are initially disabled (unchecked) to allow normal
            full-range editing operations.
        """
        super().__init__("mask lightness")
        #self.setFrameShape(QFrame.StyledPanel)

        self.controller = _controller

        self.hbox = QHBoxLayout()
        self.setLayout(self.hbox)

        self.checkboxShadows = QCheckBox("shadows")
        self.checkboxShadows.setChecked(False)
        self.checkboxBlacks = QCheckBox("blacks")
        self.checkboxBlacks.setChecked(False)
        self.checkboxMediums = QCheckBox("mediums")
        self.checkboxMediums.setChecked(False)
        self.checkboxWhites = QCheckBox("whites")
        self.checkboxWhites.setChecked(False)
        self.checkboxHighlights = QCheckBox("highlights")
        self.checkboxHighlights.setChecked(False)

        self.checkboxShadows.toggled.connect(self.clickShadows)
        self.checkboxBlacks.toggled.connect(self.clickBlacks)
        self.checkboxMediums.toggled.connect(self.clickMediums)
        self.checkboxWhites.toggled.connect(self.clickWhites)
        self.checkboxHighlights.toggled.connect(self.clickHighlights)

        self.hbox.addWidget(self.checkboxShadows)
        self.hbox.addWidget(self.checkboxBlacks)
        self.hbox.addWidget(self.checkboxMediums)
        self.hbox.addWidget(self.checkboxWhites)
        self.hbox.addWidget(self.checkboxHighlights)

    # callbacks
    def clickShadows(self):
        """
        Handle shadows mask checkbox toggle.
        
        Activates or deactivates masking for the shadows tone range (0-20%)
        when the shadows checkbox state changes. Enables selective editing
        of the darkest image regions.
        """
        if self.controller.callBackActive:  
            self.controller.maskChange("shadows", self.checkboxShadows.isChecked())
            
    def clickBlacks(self):
        """
        Handle blacks mask checkbox toggle.
        
        Activates or deactivates masking for the blacks tone range (20-40%)
        when the blacks checkbox state changes. Enables selective editing
        of dark mid-tone regions.
        """
        if self.controller.callBackActive:  
            self.controller.maskChange("blacks", self.checkboxBlacks.isChecked())
            
    def clickMediums(self):
        """
        Handle mediums mask checkbox toggle.
        
        Activates or deactivates masking for the mediums tone range (40-60%)
        when the mediums checkbox state changes. Enables selective editing
        of mid-tone regions including skin tones and neutral greys.
        """
        if self.controller.callBackActive:  
            self.controller.maskChange("mediums", self.checkboxMediums.isChecked())
            
    def clickWhites(self):
        """
        Handle whites mask checkbox toggle.
        
        Activates or deactivates masking for the whites tone range (60-80%)
        when the whites checkbox state changes. Enables selective editing
        of bright mid-tone regions and highlight transitions.
        """
        if self.controller.callBackActive:  
            self.controller.maskChange("whites", self.checkboxWhites.isChecked())
            
    def clickHighlights(self):
        """
        Handle highlights mask checkbox toggle.
        
        Activates or deactivates masking for the highlights tone range (80-100%)
        when the highlights checkbox state changes. Enables selective editing
        of the brightest image regions and specular highlights.
        """
        if self.controller.callBackActive:  
            self.controller.maskChange("highlights", self.checkboxHighlights.isChecked())
# ------------------------------------------------------------------------------------------
# --- class HDRviewerView(QFrame) ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class HDRviewerView(QFrame):
    """
    HDR display preview control interface for external HDR monitor management.
    
    Provides essential controls for managing real-time HDR image preview on external
    HDR-capable displays. Enables workflow acceleration through live preview updates
    and side-by-side comparison capabilities for professional HDR image editing.
    
    HDR Display Features:
    - Real-time preview: Live updates of edited HDR images on external HDR displays
    - Automatic updates: Optional real-time synchronization with editing parameters
    - Raw/edited comparison: Side-by-side visualization of original vs processed images
    - Display reset: Clear HDR display and return to neutral state
    - Manual updates: On-demand refresh for selective preview control
    
    Workflow Integration:
    The HDR viewer integrates seamlessly with the editing workflow by providing
    immediate visual feedback on actual HDR hardware. This allows colorists and
    HDR specialists to:
    
    - Evaluate tone mapping accuracy on target display technology
    - Assess color gamut utilization and saturation levels
    - Verify highlight detail preservation and shadow lifting
    - Compare multiple editing iterations efficiently
    - Validate HDR content for delivery and mastering
    
    Display Requirements:
    Requires external HDR-capable monitor configured through preferences system.
    Supports various HDR display standards and peak brightness levels based on
    hardware capabilities and preference configuration.
    
    Layout Structure:
    +------------------------------------------+
    | hdr preview                    [reset]   |  <- Top row
    +------------------------------------------+
    | [✓] auto  [update]  [compare]           |  <- Bottom row
    +------------------------------------------+
    
    Control Functions:
    - Auto checkbox: Enable/disable automatic preview updates during editing
    - Update button: Manual refresh of HDR display with current image state
    - Compare button: Side-by-side display of raw and edited versions
    - Reset button: Clear HDR display and show default splash screen
    
    Performance Considerations:
    Auto-update mode may impact editing performance due to continuous HDR image
    processing and display updates. Manual update mode provides better responsiveness
    for complex editing operations while maintaining preview capability.
    
    Attributes:
        - controller (HDRController): Parent controller managing HDR display hardware
        - label (QLabel): "hdr preview" identifier label
        - resetButton (QPushButton): Clear HDR display control
        - updateButton (QPushButton): Manual HDR preview refresh
        - compareButton (QPushButton): Raw/edited comparison display
        - autoCheckBox (QCheckBox): Automatic update toggle
        - vbox (QVBoxLayout): Main vertical layout container
        - hboxUp, hboxDown (QHBoxLayout): Top and bottom control row layouts
        - hboxUpContainer, hboxDownContainer (QFrame): Layout containers

    Usage Example:
        # Enable auto-preview for real-time editing feedback
        hdr_viewer.autoCheckBox.setChecked(True)
        
        # Manual update after complex editing operations
        hdr_viewer.updateButton.click()
        
        # Compare original and edited versions
        hdr_viewer.compareButton.click()
    
    Integration Notes:
        - Works with AppView menu shortcuts (Ctrl+H, Ctrl+C, Ctrl+K)
        - Synchronized with EditImageController editing operations
        - Respects HDR display preferences and configuration
        - Supports multiple HDR display standards and devices
    """
    def __init__(self, _controller= None, build = False):
        """
        Initialize HDR preview control interface with display management capabilities.
        
        Creates the HDR viewer interface with auto-update checkbox, manual control
        buttons, and proper layout arrangement. Optionally restores previous auto-update
        state when rebuilding the interface.
        
        Args:
            _controller (HDRController, optional): Parent HDR display controller
                managing external HDR hardware and display operations
            build (bool): Whether to restore previous auto-update state from
                existing model. Used when rebuilding interface during dock switching.
                
        Interface Setup:
            - Top row: "hdr preview" label + reset button
            - Bottom row: auto checkbox + update button + compare button
            - All controls connected to appropriate controller callbacks
            
        State Restoration:
            When build=True, retrieves previous autoPreviewHDR setting from
            the editing model to maintain consistent user preferences across
            interface rebuilds and dock panel switches.
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)

        self.controller = _controller

        self.vbox = QVBoxLayout()
        self.hboxUp = QHBoxLayout()
        self.hboxDown = QHBoxLayout()

        self.label= QLabel("hdr preview")
        self.resetButton = QPushButton("reset")
        self.updateButton = QPushButton("update")
        self.compareButton = QPushButton("compare")
        self.autoCheckBox = QCheckBox("auto")

        if build:
            cValue = self.controller.parent.view.dock.view.childControllers[0].model.autoPreviewHDR
            self.autoCheckBox.setChecked(cValue)
        else:
            self.autoCheckBox.setChecked(False)

        self.hboxUpContainer = QFrame()
        self.hboxUpContainer.setLayout(self.hboxUp)
        self.hboxUp.addWidget(self.label)
        self.hboxUp.addWidget(self.resetButton)

        self.hboxDownContainer = QFrame()
        self.hboxDownContainer.setLayout(self.hboxDown)
        self.hboxDown.addWidget(self.autoCheckBox)
        self.hboxDown.addWidget(self.updateButton)
        self.hboxDown.addWidget(self.compareButton)

        self.vbox.addWidget(self.hboxUpContainer)
        self.vbox.addWidget(self.hboxDownContainer)

        self.setLayout(self.vbox)

        self.resetButton.clicked.connect(self.reset)
        self.updateButton.clicked.connect(self.update)
        self.compareButton.clicked.connect(self.compare)
        self.autoCheckBox.toggled.connect(self.auto)

    def reset(self):
        """
        Clear HDR display and show default splash screen.
        
        Resets the external HDR monitor to a neutral state by displaying
        the default splash screen. Useful for clearing previous previews
        and returning to a known baseline state.
        """
        self.controller.displaySplash()

    def update(self):
        """
        Manually refresh HDR display with current image state.
        
        Triggers immediate update of the external HDR display with the
        currently processed image, reflecting all current editing parameters.
        Provides on-demand preview control for performance optimization.
        """
        self.controller.callBackUpdate()

    def compare(self):
        """
        Display side-by-side raw and edited image comparison.
        
        Shows both the original unprocessed image and the current edited
        version simultaneously on the HDR display for direct visual
        comparison. Essential for evaluating editing effectiveness and
        maintaining reference to original image characteristics.
        """
        self.controller.callBackCompare()

    def auto(self):
        """
        Toggle automatic HDR preview updates based on checkbox state.
        
        Enables or disables real-time HDR display updates that synchronize
        automatically with editing parameter changes. When enabled, provides
        immediate visual feedback but may impact editing performance.
        """
        self.controller.callBackAuto(self.autoCheckBox.isChecked())
# ------------------------------------------------------------------------------------------
# --- class LchColorSelectorView(QFrame) ---------------------------------------------------
# ------------------------------------------------------------------------------------------
class LchColorSelectorView(QFrame):
    """
    LCH color space selection and editing interface.
    
    Provides comprehensive color editing in LCH (Lightness, Chroma, Hue) space
    with visual color bars, range selection sliders, and targeted editing controls.
    Allows precise color selection and adjustment with real-time visual feedback.
    
    Color Selection Components:
    - Hue bar: Visual hue spectrum (0-360°)
    - Chroma bar: Saturation/colorfulness visualization (0-100)
    - Lightness bar: Brightness visualization (0-100)
    - Range sliders: Min/max selection for each component
    
    Color Editing Components:
    - Hue shift: Rotate colors around hue wheel (-180° to +180°)
    - Exposure: Brightness adjustment (-3 to +3 stops)
    - Contrast: Local contrast enhancement (-100 to +100)
    - Saturation: Colorfulness adjustment (-100 to +100)
    
    Additional Features:
    - Selection mask preview
    - Reset buttons for selection and editing parameters
    - Real-time color bar updates based on selection
    
    Layout Structure:
    +------------------------------------------+
    | Hue Chroma Lightness color selector      |
    +------------------------------------------+
    | [Hue Color Bar Visualization]           |
    | [Min Hue Slider] [Max Hue Slider]       |
    | [Selected Hue Range Bar]                |
    +------------------------------------------+
    | [Chroma Color Bar Visualization]        |
    | [Min Chroma Slider] [Max Chroma Slider] |
    +------------------------------------------+
    | [Lightness Color Bar Visualization]     |
    | [Min Light Slider] [Max Light Slider]   |
    +------------------------------------------+
    | [Reset Selection]                        |
    +------------------------------------------+
    | color editor: hue shift, exposure, contrast, saturation |
    | hue shift:  [slider] [value]            |
    | saturation: [slider] [value]            |
    | exposure:   [slider] [value]            |
    | contrast:   [slider] [value]            |
    | [x] show selection                       |
    | [Reset Edit]                             |
    +------------------------------------------+
    
    Attributes:
        - controller (LchColorSelectorController): Parent controller reference
        - imageHueController, imageSaturationController, imageLightnessController: Color bar displays
        - imageHueRangeController: Selected hue range visualization
        - sliderHueMin, sliderHueMax: Hue range selection sliders
        - sliderChromaMin, sliderChromaMax: Chroma range selection sliders
        - sliderLightMin, sliderLightMax: Lightness range selection sliders
        - sliderHueShift, sliderExposure, sliderContrast, sliderSaturation: Editing controls
        - valueHueShift, valueExposure, valueContrast, valueSaturation: Value displays
        - checkboxMask: Selection mask preview toggle
        - resetSelection, resetEdit: Reset buttons
    """
    def __init__(self, _controller, defaultValues=None):
        """
        Initialize LCH color selector interface.
        
        Args:
            _controller (LchColorSelectorController): Parent controller instance
            defaultValues: Optional default parameter values
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = _controller

        self.vbox = QVBoxLayout()

        self.labelSelector = QLabel("Hue Chroma Lighness color selector")
        # procedural image: Hue bar
        hueBarLch = hdrCore.image.Image.buildLchColorData((75,75), (100,100), (0,360), (20,720), width='h', height='c')
        hueBarRGB = hdrCore.processing.Lch_to_sRGB(hueBarLch,apply_cctf_encoding=True, clip=True)
        self.imageHueController = controller.ImageWidgetController()
        self.imageHueController.view.setMinimumSize(2, 72)
        self.imageHueController.setImage(hueBarRGB)
        hueBar2Lch = hdrCore.image.Image.buildLchColorData((75,75), (100,100), (0,360), (20,720), width='h', height='c')
        hueBar2RGB = hdrCore.processing.Lch_to_sRGB(hueBar2Lch,apply_cctf_encoding=True, clip=True)
        self.imageHueRangeController = controller.ImageWidgetController()
        self.imageHueRangeController.view.setMinimumSize(2, 72)
        self.imageHueRangeController.setImage(hueBarRGB)
        # slider min
        self.sliderHueMin = QSlider(Qt.Horizontal)
        self.sliderHueMin.setRange(0,360)
        self.sliderHueMin.setValue(0)
        self.sliderHueMin.setSingleStep(1)
        # slider max
        self.sliderHueMax = QSlider(Qt.Horizontal)
        self.sliderHueMax.setRange(0,360)
        self.sliderHueMax.setValue(360)
        self.sliderHueMax.setSingleStep(1)

        # procedural image: Saturation bar
        saturationBarLch = hdrCore.image.Image.buildLchColorData((75,75), (0,100), (180,180), (20,720), width='c', height='L')
        saturationBarRGB = hdrCore.processing.Lch_to_sRGB(saturationBarLch,apply_cctf_encoding=True, clip=True)
        self.imageSaturationController = controller.ImageWidgetController()
        self.imageSaturationController.view.setMinimumSize(2, 72)
        self.imageSaturationController.setImage(saturationBarRGB)
        # slider min
        self.sliderChromaMin = QSlider(Qt.Horizontal)
        self.sliderChromaMin.setRange(0,100)
        self.sliderChromaMin.setValue(0)
        self.sliderChromaMin.setSingleStep(1)
        # slider max
        self.sliderChromaMax = QSlider(Qt.Horizontal)
        self.sliderChromaMax.setRange(0,100)
        self.sliderChromaMax.setValue(100)
        self.sliderChromaMax.setSingleStep(1)

        # procedural image:lightness bar
        lightnessBarLch = hdrCore.image.Image.buildLchColorData((0,100), (0,0), (180,180), (20,720), width='L', height='c')
        lightnessBarRGB = hdrCore.processing.Lch_to_sRGB(lightnessBarLch,apply_cctf_encoding=True, clip=True)
        self.imageLightnessController = controller.ImageWidgetController()
        self.imageLightnessController.view.setMinimumSize(2, 72)
        self.imageLightnessController.setImage(lightnessBarRGB)
        # slider min
        self.sliderLightMin = QSlider(Qt.Horizontal)
        self.sliderLightMin.setRange(0,300)
        self.sliderLightMin.setValue(0)
        self.sliderLightMin.setSingleStep(1)        
        # slider max
        self.sliderLightMax = QSlider(Qt.Horizontal)
        self.sliderLightMax.setRange(0,300)
        self.sliderLightMax.setValue(300)
        self.sliderLightMax.setSingleStep(1)

        # editor
        self.labelEditor =      QLabel("color editor: hue shift, exposure, contrast, saturation")

        # hue shift [-180,+180]
        self.frameHueShift = QFrame()
        self.layoutHueShift = QHBoxLayout()
        self.frameHueShift.setLayout(self.layoutHueShift)
        self.sliderHueShift =   QSlider(Qt.Horizontal)
        self.sliderHueShift.setRange(-180,+180)
        self.sliderHueShift.setValue(0)
        self.sliderHueShift.setSingleStep(1) 
        self.valueHueShift = QLineEdit()
        self.valueHueShift.setText(str(0.0))  
        self.layoutHueShift.addWidget(QLabel("hue shift"))
        self.layoutHueShift.addWidget(self.sliderHueShift)
        self.layoutHueShift.addWidget(self.valueHueShift)

        # exposure [-3 , +3]
        self.frameExposure = QFrame()
        self.layoutExposure = QHBoxLayout()
        self.frameExposure.setLayout(self.layoutExposure)
        self.sliderExposure =   QSlider(Qt.Horizontal)
        self.sliderExposure.setRange(-90,+90)
        self.sliderExposure.setValue(0)
        self.sliderExposure.setSingleStep(1) 
        self.valueExposure = QLineEdit()
        self.valueExposure.setText(str(0.0))  
        self.layoutExposure.addWidget(QLabel("exposure"))
        self.layoutExposure.addWidget(self.sliderExposure)
        self.layoutExposure.addWidget(self.valueExposure)

        # contrast [-100 , +100]
        self.frameContrast = QFrame()
        self.layoutContrast = QHBoxLayout()
        self.frameContrast.setLayout(self.layoutContrast)
        self.sliderContrast =   QSlider(Qt.Horizontal)
        self.sliderContrast.setRange(-100,+100)
        self.sliderContrast.setValue(0)
        self.sliderContrast.setSingleStep(1) 
        self.valueContrast = QLineEdit()
        self.valueContrast.setText(str(0.0))  
        self.layoutContrast.addWidget(QLabel("contrast"))
        self.layoutContrast.addWidget(self.sliderContrast)
        self.layoutContrast.addWidget(self.valueContrast)

        # saturation [-100 , +100]
        self.frameSaturation = QFrame()
        self.layoutSaturation = QHBoxLayout()
        self.frameSaturation.setLayout(self.layoutSaturation)
        self.sliderSaturation =   QSlider(Qt.Horizontal)
        self.sliderSaturation.setRange(-100,+100)
        self.sliderSaturation.setValue(0)
        self.sliderSaturation.setSingleStep(1) 
        self.valueSaturation = QLineEdit()
        self.valueSaturation.setText(str(0.0))  
        self.layoutSaturation.addWidget(QLabel("saturation"))
        self.layoutSaturation.addWidget(self.sliderSaturation)
        self.layoutSaturation.addWidget(self.valueSaturation)

        # -----
        self.resetSelection  = QPushButton("reset selection")
        self.resetEdit  = QPushButton("reset edit")
        # -----

        # mask
        self.checkboxMask = QCheckBox("show selection")
        self.checkboxMask.setChecked(False)

        self.vbox.addWidget(self.labelSelector)

        self.vbox.addWidget(self.imageHueController.view)
        self.vbox.addWidget(self.sliderHueMin)
        self.vbox.addWidget(self.sliderHueMax)
        self.vbox.addWidget(self.imageHueRangeController.view)


        self.vbox.addWidget(self.imageSaturationController.view)
        self.vbox.addWidget(self.sliderChromaMin)
        self.vbox.addWidget(self.sliderChromaMax)

        self.vbox.addWidget(self.imageLightnessController.view)
        self.vbox.addWidget(self.sliderLightMin)
        self.vbox.addWidget(self.sliderLightMax)

        # -----
        self.vbox.addWidget(self.resetSelection)
        # -----


        self.vbox.addWidget(self.labelEditor)
        self.vbox.addWidget(self.frameHueShift)
        self.vbox.addWidget(self.frameSaturation)
        self.vbox.addWidget(self.frameExposure)
        self.vbox.addWidget(self.frameContrast)

        self.vbox.addWidget(self.checkboxMask)
        # -----
        self.vbox.addWidget(self.resetEdit)
        # -----
        self.setLayout(self.vbox)

        # callbacks  
        self.sliderHueMin.valueChanged.connect(self.sliderHueChange)
        self.sliderHueMax.valueChanged.connect(self.sliderHueChange)
        self.sliderChromaMin.valueChanged.connect(self.sliderChromaChange)
        self.sliderChromaMax.valueChanged.connect(self.sliderChromaChange)
        self.sliderLightMin.valueChanged.connect(self.sliderLightnessChange)
        self.sliderLightMax.valueChanged.connect(self.sliderLightnessChange)
        self.sliderExposure.valueChanged.connect(self.sliderExposureChange)
        self.sliderSaturation.valueChanged.connect(self.sliderSaturationChange)
        self.sliderContrast.valueChanged.connect(self.sliderContrastChange)
        self.sliderHueShift.valueChanged.connect(self.sliderHueShiftChange)
        self.checkboxMask.toggled.connect(self.checkboxMaskChange)
        # -----
        self.resetSelection.clicked.connect(self.controller.resetSelection)
        self.resetEdit.clicked.connect(self.controller.resetEdit)


    # callbacks
    def sliderHueChange(self):
        """
        Handle hue range selection changes and update visual feedback.
        
        Updates hue range visualization and chroma bar based on selected
        hue range, then notifies controller of the change.
        """
        hmin = self.sliderHueMin.value()
        hmax = self.sliderHueMax.value()

        # redraw hue range and chroma bar
        hueRangeBarLch = hdrCore.image.Image.buildLchColorData((75,75), (100,100), (hmin,hmax), (20,720), width='h', height='c')
        hueRangeBarRGB = hdrCore.processing.Lch_to_sRGB(hueRangeBarLch,apply_cctf_encoding=True, clip=True)
        self.imageHueRangeController.setImage(hueRangeBarRGB)
        saturationBarLch = hdrCore.image.Image.buildLchColorData((75,75), (0,100), (hmin,hmax), (20,720), width='c', height='L')
        saturationBarRGB = hdrCore.processing.Lch_to_sRGB(saturationBarLch,apply_cctf_encoding=True, clip=True)
        self.imageSaturationController.setImage(saturationBarRGB)

        # call controller
        self.controller.sliderHueChange(hmin,hmax)

    def sliderChromaChange(self):
        """Handle chroma range selection changes."""
        vmin = self.sliderChromaMin.value()
        vmax = self.sliderChromaMax.value()
        # call controller
        self.controller.sliderChromaChange(vmin,vmax)

    def sliderLightnessChange(self):
        """Handle lightness range selection changes."""
        vmin = self.sliderLightMin.value()/3.0
        vmax = self.sliderLightMax.value()/3.0
        # call controller
        self.controller.sliderLightnessChange(vmin,vmax)

    def sliderExposureChange(self):
        """Handle exposure editing changes with 0.1 stop precision."""
        ev = round(self.sliderExposure.value()/30,1)
        # force to 0.1 precision
        self.valueExposure.setText(str(ev))
        self.controller.sliderExposureChange(ev)

    def sliderSaturationChange(self):
        """Handle saturation editing changes."""
        ev = self.sliderSaturation.value()
        self.valueSaturation.setText(str(ev))
        self.controller.sliderSaturationChange(ev)

    def sliderContrastChange(self):
        """Handle contrast editing changes."""
        ev = self.sliderContrast.value()
        self.valueContrast.setText(str(ev))
        self.controller.sliderContrastChange(ev)

    def sliderHueShiftChange(self):
        """Handle hue shift editing changes."""
        hs = self.sliderHueShift.value()
        self.valueHueShift.setText(str(hs))
        self.controller.sliderHueShiftChange(hs)

    def checkboxMaskChange(self):
        """Handle selection mask preview toggle."""
        self.controller.checkboxMaskChange(self.checkboxMask.isChecked())
# ------------------------------------------------------------------------------------------
# --- class GeometryView(QFrame) -----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class GeometryView(QFrame):
    """
    Geometric transformation controls for rotation and cropping adjustments.
    
    Provides sliders for basic geometric corrections including:
    - Vertical cropping adjustment: Fine-tune vertical positioning for aspect ratio cropping
    - Rotation: Small angle corrections for straightening images
    
    Layout:
    +------------------------------------------+
    | cropping adj. [slider] [value]          |
    | rotation      [slider] [value]          |
    +------------------------------------------+
    
    Attributes:
        - controller (GeometryController): Parent controller reference
        - sliderCroppingVerticalAdjustement: Vertical cropping offset slider
        - valueCroppingVerticalAdjustement: Cropping adjustment value display
        - sliderRotation: Rotation angle slider
        - valueRotation: Rotation value display
    """
    def __init__(self, _controller):
        """
        Initialize geometry transformation interface.
        
        Args:
            _controller (GeometryController): Parent controller instance
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = _controller

        self.vbox = QVBoxLayout()

        # cropping adjustement 
        self.frameCroppingVerticalAdjustement = QFrame()
        self.layoutCroppingVerticalAdjustement = QHBoxLayout()
        self.frameCroppingVerticalAdjustement.setLayout(self.layoutCroppingVerticalAdjustement)
        self.sliderCroppingVerticalAdjustement =   QSlider(Qt.Horizontal)
        self.sliderCroppingVerticalAdjustement.setRange(-100,+100)
        self.sliderCroppingVerticalAdjustement.setValue(0)
        self.sliderCroppingVerticalAdjustement.setSingleStep(1) 
        self.valueCroppingVerticalAdjustement = QLineEdit()
        self.valueCroppingVerticalAdjustement.setText(str(0.0))  
        self.layoutCroppingVerticalAdjustement.addWidget(QLabel("cropping adj."))
        self.layoutCroppingVerticalAdjustement.addWidget(self.sliderCroppingVerticalAdjustement)
        self.layoutCroppingVerticalAdjustement.addWidget(self.valueCroppingVerticalAdjustement)

        # rotation 
        self.frameRotation = QFrame()
        self.layoutRotation = QHBoxLayout()
        self.frameRotation.setLayout(self.layoutRotation)
        self.sliderRotation =   QSlider(Qt.Horizontal)
        self.sliderRotation.setRange(-60,+60)
        self.sliderRotation.setValue(0)
        self.sliderRotation.setSingleStep(1) 
        self.valueRotation = QLineEdit()
        self.valueRotation.setText(str(0.0))  
        self.layoutRotation.addWidget(QLabel("rotation"))
        self.layoutRotation.addWidget(self.sliderRotation)
        self.layoutRotation.addWidget(self.valueRotation)

        self.vbox.addWidget(self.frameCroppingVerticalAdjustement)
        self.vbox.addWidget(self.frameRotation)

        self.setLayout(self.vbox)

        self.sliderCroppingVerticalAdjustement.valueChanged.connect(self.sliderCroppingVerticalAdjustementChange)
        self.sliderRotation.valueChanged.connect(self.sliderRotationChange)

    # callbacks
    def sliderCroppingVerticalAdjustementChange(self):
        """Handle vertical cropping adjustment changes."""
        v = self.sliderCroppingVerticalAdjustement.value()
        self.valueCroppingVerticalAdjustement.setText(str(v))
        # call controller
        self.controller.sliderCroppingVerticalAdjustementChange(v)

    def sliderRotationChange(self):
        """Handle rotation angle changes with 1/6 degree precision."""
        v = self.sliderRotation.value()/6
        self.valueRotation.setText(str(v))
        # call controller
        self.controller.sliderRotationChange(v)
# ------------------------------------------------------------------------------------------
# --- class ImageAestheticsView(QSplitter) -------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageAestheticsView(QSplitter):
    """
    Image aesthetics analysis and color palette visualization interface.
    
    Provides tools for analyzing image aesthetics including dominant color
    extraction, color palette generation, and visual composition analysis.
    Uses K-means clustering for color palette extraction with configurable
    parameters.
    
    Layout Structure:
    - Top: Image preview showing current processing result
    - Bottom: Scrollable controls panel with:
      * Process output selector (which processing step to analyze)
      * Number of colors configuration (2-8 colors)
      * Generated color palette visualization
    
    Analysis Features:
    - Dominant color extraction using K-means clustering
    - Configurable color count (2-8 dominant colors)
    - Process step selection for analysis input
    - Visual color palette representation
    - Extensible for composition and strength line analysis
    
    Layout:
    +------------------------------------------+
    | [Image Preview]                          |
    +------------------------------------------+
    | color palette                            |
    | process output: [dropdown]               |
    | number of colors: [spinbox]              |
    | [Color Palette Visualization]           |
    +------------------------------------------+
    
    Attributes:
        - controller (ImageAestheticsController): Parent controller reference
        - imageWidgetController (ImageWidgetController): Main image display
        - labelColorPalette: Color palette section label
        - labelNodeSelector: Process output selection label
        - nodeSelector (QComboBox): Process step selection dropdown
        - labelColorsNumber: Color count configuration label
        - nbColors (QSpinBox): Number of colors to extract (2-8)
        - paletteImageWidgetController (ImageWidgetController): Color palette display
        - scroll (QScrollArea): Scrollable container for controls
    """
    def __init__(self, _controller, build=False):
        """
        Initialize image aesthetics analysis interface.
        
        Args:
            _controller (ImageAestheticsController): Parent controller instance
            build (bool): Whether to restore previous state during construction
        """
        if pref.verbose: print(" [VIEW] >> AestheticsImageView.__init__(",")")
        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QVBoxLayout()

        # --------------- color palette: node selector(node name), color number, palette image.
        self.labelColorPalette = QLabel("color palette")
        self.labelNodeSelector = QLabel("process output:")
        self.nodeSelector = QComboBox(self)

        # recover process nodes names from buildProcessPipe
        processNodeNameList = []
        emptyProcessPipe = model.EditImageModel.buildProcessPipe()
        for node in emptyProcessPipe.processNodes: processNodeNameList.append(node.name)

        # add 'output' at the end to help user
        processNodeNameList.append('output')

        self.nodeSelector.addItems(processNodeNameList)
        self.nodeSelector.setCurrentIndex(len(processNodeNameList)-1)

        # QSpinBox
        self.labelColorsNumber = QLabel("number of colors:")
        self.nbColors = QSpinBox(self)
        self.nbColors.setRange(2,8)
        self.nbColors.setValue(5)

        self.paletteImageWidgetController = controller.ImageWidgetController()
        imgPalette = hdrCore.aesthetics.Palette('defaultLab5',np.linspace([0,0,0],[100,0,0],5),hdrCore.image.ColorSpace.build('Lab'), hdrCore.image.imageType.SDR).createImageOfPalette()
        self.paletteImageWidgetController.setImage(imgPalette)
        self.paletteImageWidgetController.view.setMinimumSize(40, 10)

        # add widgets to layout
        self.layout.addWidget(self.labelColorPalette)
        self.layout.addWidget(self.labelNodeSelector)
        self.layout.addWidget(self.nodeSelector)
        self.layout.addWidget(self.labelColorsNumber)
        self.layout.addWidget(self.nbColors)
        self.layout.addWidget(self.paletteImageWidgetController.view)

        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        # scroll and etc.
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        # add widget to QSplitter
        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60,40])
        # --------------- composition:
        # --------------- strength line:

    def setProcessPipe(self,processPipe, paletteImg):
        """
        Update display with new ProcessPipe and color palette.
        
        Args:
            processPipe (ProcessPipe): New processing pipeline to display
            paletteImg (numpy.ndarray): Generated color palette visualization
        """
        self.imageWidgetController.setImage(processPipe.getImage())
        self.paletteImageWidgetController.setImage(paletteImg)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ColorEditorsAutoView(QPushButton):
    """
    Automatic color editor configuration button.
    
    Provides a single button interface for automatically configuring
    multiple color editors based on K-means analysis of image colors.
    When activated, analyzes the current image to extract dominant colors
    and sets up color editors with appropriate selection ranges.
    
    Warning: Activating this function will reset all existing color
    editor configurations to the automatically generated settings.
    
    Attributes:
        - controller (ColorEditorsAutoController): Parent controller reference
    """
    def __init__(self,controller):
        """
        Initialize automatic color editor configuration button.
        
        Args:
            controller (ColorEditorsAutoController): Parent controller instance
        """
        super().__init__("auto color selection [! reset edit]")
        self.controller = controller

        self.clicked.connect(self.controller.auto)
# ------------------------------------------------------------------------------------------

