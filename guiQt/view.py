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
    AppView: Main application window with menus and docking
    ImageGalleryView: Grid-based image gallery with pagination
    EditImageView: Complete HDR editing interface
    ImageInfoView: Image metadata and information display
    ToneCurveView: B-spline tone curve editing with matplotlib
    LchColorSelectorView: LCH color space selection interface
    HDRviewerView: HDR display controls and preview
    ImageAestheticsView: Color palette and aesthetics visualization

Widget Utilities:
    ImageWidgetView: Basic image display widget
    FigureWidget: Matplotlib integration for curve editing
    AdvanceSliderView: Enhanced slider with auto/reset functionality
    AdvanceLineEdit: Labeled line edit widget
    AdvanceCheckBox: Labeled checkbox widget

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
        controller: Reference to controlling ImageWidgetController
        label (QLabel): Qt label widget for pixmap display
        imagePixmap (QPixmap): Current pixmap for display
    
    Methods:
        resize: Update widget and pixmap scaling
        resizeEvent: Handle Qt resize events
        setPixmap: Set image from numpy array with preprocessing
        setQPixmap: Set pre-processed QPixmap directly
        emptyImageColorData: Generate default placeholder image [Static]
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
        fig (Figure): Matplotlib figure object
        axes: Matplotlib axes for plotting
    
    Methods:
        plot: Draw line plot with specified data and style
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
        controller: Reference to ImageGalleryController
        shapeMode: Current grid layout mode (GalleryMode enum)
        pageNumber (int): Current page index (0-based)
        imagesControllers (list): List of ImageWidgetController instances
        images (QFrame): Frame containing image grid
        imagesLayout (QGridLayout): Grid layout for images
        buttons (QWidget): Navigation button container
        pageNumberLabel (QLabel): Current page display
    
    Methods:
        currentPage: Get current page number
        changePageNumber: Navigate to different page
        updateImages: Refresh image display for current page
        updateImage: Update specific image in grid
        resetGridLayoutWidgets: Clear current grid widgets
        buildGridLayoutWidgets: Create new grid layout
        wheelEvent: Handle mouse wheel navigation
        mousePressEvent: Handle image selection clicks
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
        controller (AppController): Parent application controller
        imageGalleryController (ImageGalleryController): Central gallery management
        dock (MultiDockController): Right panel controller
        topContainer (QWidget): Central widget container
        menuExport (QAction): Export menu action reference
        menuExportAll (QAction): Export all menu action reference
    
    Methods:
        getImageGalleryController: Access gallery controller
        resizeEvent: Handle window resize events
        setWindowGeometry: Configure window size and position
        buildFileMenu: Create file operations menu
        buildPreferences: Create HDR display preferences menu
        buildDisplayHDR: Create HDR display menu
        buildExport: Create export functionality menu
        buildDockMenu: Create dock panel switching menu
        closeEvent: Handle application shutdown
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
        controller (ImageInfoController): Parent controller reference
        imageWidgetController (ImageWidgetController): Image preview controller
        layout (QFormLayout): Metadata fields layout
        imageName, imagePath, imageSize, etc.: Technical metadata widgets
        userDefinedTags (list[AdvanceCheckBox]): Custom metadata checkboxes
        scroll (QScrollArea): Scrollable container for metadata fields
    
    Methods:
        setProcessPipe: Update display with new image and metadata
        metadataChange: Handle user changes to custom metadata
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
        label (QLabel): Display label for the field
        lineEdit (QLineEdit): Text input widget
    
    Methods:
        setText: Update the displayed text value
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
        parent (ImageInfoView): Parent view for callback delegation
        leftText (str): Left label text (group name)
        rightText (str): Right label text (tag name)
        label (QLabel): Display label widget
        checkbox (QCheckBox): Checkbox control
    
    Methods:
        setState: Set checkbox checked state
        toggled: Handle checkbox state changes
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
        controller (EditImageController): Parent controller reference
        imageWidgetController (ImageWidgetController): Image preview controller
        exposure, contrast, saturation (AdvanceSliderController): Basic adjustment controls
        tonecurve (ToneCurveController): B-spline tone curve editor
        lightnessmask (LightnessMaskController): Tone range masking
        colorEditor0-4 (LchColorSelectorController): Independent color editors
        colorEditorsAuto (ColorEditorsAutoController): Automatic color selection
        geometry (GeometryController): Rotation and cropping controls
        hdrPreview (HDRviewerView): HDR display preview interface
        scroll (QScrollArea): Scrollable container for all controls
    
    Methods:
        setImage: Update image preview display
        autoExposure: Trigger automatic exposure calculation
        changeExposure: Handle manual exposure adjustments
        autoContrast: Trigger automatic contrast calculation
        changeContrast: Handle manual contrast adjustments
        autoSaturation: Trigger automatic saturation calculation
        changeSaturation: Handle manual saturation adjustments
        plotToneCurve: Update tone curve visualization
        setProcessPipe: Initialize interface with ProcessPipe parameters
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
    def __init__(self, _controller, HDRcontroller=None):
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
            change active dock
            nb = 0 > editing imag dock
            nb = 1 > image info and metadata dock
            nb = 2 > image aesthetics model 
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
        if pref.verbose:  print(" [VIEW] >> MultiDockView.setProcessPipe(",processPipe.getImage().name,")")
        return self.childController.setProcessPipe(processPipe)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# --- class AdvanceSliderView(QFrame) ------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceSliderView(QFrame):
    def __init__(self, controller, name,defaultValue, range, step):
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

    def __init__(self, controller):
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
        if self.controller.callBackActive:
            value = self.sliderShadows.value()
            self.controller.sliderChange("shadows", value)
        pass

    def sliderBlacksChange(self):
        if self.controller.callBackActive:
            value = self.sliderBlacks.value()
            self.controller.sliderChange("blacks", value)
        pass

    def sliderMediumsChange(self):
        if self.controller.callBackActive:
            value = self.sliderMediums.value()
            self.controller.sliderChange("mediums", value)
        pass

    def sliderWhitesChange(self):
        if self.controller.callBackActive:
            value = self.sliderWhites.value()
            self.controller.sliderChange("whites", value)
        pass

    def sliderHighlightsChange(self):
        if self.controller.callBackActive:
            value = self.sliderHighlights.value()
            self.controller.sliderChange("highlights", value)
        pass

    def resetShadowsCB(self):
        if self.controller.callBackActive: self.controller.reset("shadows")

    def resetBlacksCB(self):
        if self.controller.callBackActive: self.controller.reset("blacks")
    
    def resetMediumsCB(self):
        if self.controller.callBackActive: self.controller.reset("mediums")

    def resetWhitesCB(self):
        if self.controller.callBackActive: self.controller.reset("whites")

    def resetHighlightsCB(self):
        if self.controller.callBackActive: self.controller.reset("highlights")
# ------------------------------------------------------------------------------------------
# --- class LightnessMaskView(QGroupBox) ---------------------------------------------------
# ------------------------------------------------------------------------------------------
class LightnessMaskView(QGroupBox):
    def __init__(self, _controller):
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
        if self.controller.callBackActive:  self.controller.maskChange("shadows", self.checkboxShadows.isChecked())
    def clickBlacks(self):     
        if self.controller.callBackActive:  self.controller.maskChange("blacks", self.checkboxBlacks.isChecked())
    def clickMediums(self):     
        if self.controller.callBackActive:  self.controller.maskChange("mediums", self.checkboxMediums.isChecked())
    def clickWhites(self):     
        if self.controller.callBackActive:  self.controller.maskChange("whites", self.checkboxWhites.isChecked())
    def clickHighlights(self):     
        if self.controller.callBackActive:  self.controller.maskChange("highlights", self.checkboxHighlights.isChecked())
# ------------------------------------------------------------------------------------------
# --- class HDRviewerView(QFrame) ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class HDRviewerView(QFrame):
    def __init__(self, _controller= None, build = False):
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

    def reset(self): self.controller.displaySplash()

    def update(self): self.controller.callBackUpdate()

    def compare(self): self.controller.callBackCompare()

    def auto(self): self.controller.callBackAuto(self.autoCheckBox.isChecked())
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
        controller (LchColorSelectorController): Parent controller reference
        imageHueController, imageSaturationController, imageLightnessController: Color bar displays
        imageHueRangeController: Selected hue range visualization
        sliderHueMin, sliderHueMax: Hue range selection sliders
        sliderChromaMin, sliderChromaMax: Chroma range selection sliders
        sliderLightMin, sliderLightMax: Lightness range selection sliders
        sliderHueShift, sliderExposure, sliderContrast, sliderSaturation: Editing controls
        valueHueShift, valueExposure, valueContrast, valueSaturation: Value displays
        checkboxMask: Selection mask preview toggle
        resetSelection, resetEdit: Reset buttons
    
    Methods:
        sliderHueChange: Handle hue range selection changes
        sliderChromaChange: Handle chroma range selection changes
        sliderLightnessChange: Handle lightness range selection changes
        sliderHueShiftChange: Handle hue shift editing changes
        sliderExposureChange: Handle exposure editing changes
        sliderSaturationChange: Handle saturation editing changes
        sliderContrastChange: Handle contrast editing changes
        checkboxMaskChange: Handle mask preview toggle
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
        controller (GeometryController): Parent controller reference
        sliderCroppingVerticalAdjustement: Vertical cropping offset slider
        valueCroppingVerticalAdjustement: Cropping adjustment value display
        sliderRotation: Rotation angle slider
        valueRotation: Rotation value display
    
    Methods:
        sliderCroppingVerticalAdjustementChange: Handle cropping adjustment changes
        sliderRotationChange: Handle rotation changes
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
        controller (ImageAestheticsController): Parent controller reference
        imageWidgetController (ImageWidgetController): Main image display
        labelColorPalette: Color palette section label
        labelNodeSelector: Process output selection label
        nodeSelector (QComboBox): Process step selection dropdown
        labelColorsNumber: Color count configuration label
        nbColors (QSpinBox): Number of colors to extract (2-8)
        paletteImageWidgetController (ImageWidgetController): Color palette display
        scroll (QScrollArea): Scrollable container for controls
    
    Methods:
        setProcessPipe: Update display with new ProcessPipe and color palette
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
        controller (ColorEditorsAutoController): Parent controller reference
    
    Methods:
        Callback is automatically connected to controller.auto() method
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

