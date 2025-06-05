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
guiQt.thread module: Threading and parallel processing management for uHDR.

This module provides comprehensive threading support for uHDR's GUI application,
enabling real-time image processing, parallel computation, and responsive user
interaction. It implements several threading patterns for different use cases:

Threading Classes:
    RequestCompute: Real-time HDR editing with single-threaded sequential processing
    RunCompute: Worker thread for HDR processing pipeline execution
    RequestLoadImage: Parallel image loading and thumbnail generation
    RunLoadImage: Worker thread for individual image loading
    pCompute: Multi-threaded HDR processing with image splitting
    pRun: Worker thread for processing image splits
    cCompute: C++ accelerated HDR processing (single-threaded)
    cRun: Worker thread for C++ pipeline execution
    RequestAestheticsCompute: Aesthetics analysis threading
    RunAestheticsCompute: Worker thread for aesthetics computation

Threading Patterns:
1. Real-time Editing: Uses RequestCompute for immediate UI feedback during editing
2. Parallel Loading: Uses RequestLoadImage for efficient gallery thumbnail generation  
3. Split Processing: Uses pCompute for large image processing with progress feedback
4. C++ Acceleration: Uses cCompute for hardware-accelerated processing
5. Background Analysis: Uses RequestAestheticsCompute for non-blocking analysis

All threading classes use Qt's QThreadPool for efficient thread management
and provide callback mechanisms for result delivery and progress updates.
"""
# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import copy, time, random
import hdrCore
from . import model
from PyQt5.QtCore import QRunnable, Qt, QThreadPool
from timeit import default_timer as timer
import preferences.preferences as pref

# -----------------------------------------------------------------------------
# --- Class RequestCompute ----------------------------------------------------
# -----------------------------------------------------------------------------
class RequestCompute(object):
    """
    Real-time HDR processing coordinator for interactive image editing.
    
    Manages parallel computation of ProcessPipe operations during interactive
    editing, ensuring responsive UI while maintaining processing order. Uses
    a single dedicated thread to compute process-pipes, storing compute requests
    when user changes editing values and restarting computation when previous
    processing finishes.
    
    Key Features:
    - Single-threaded sequential processing to avoid resource conflicts
    - Request queuing for parameter changes during processing
    - Automatic restart for pending updates
    - Parent callback for result delivery
    
    Attributes:
        - parent (EditImageModel): Parent model for result callbacks
        - requestDict (dict): Stores pending parameter updates {processNodeId: params}
        - pool (QThreadPool): Qt thread pool for worker management
        - processpipe (ProcessPipe): Active processing pipeline reference
        - readyToRun (bool): Processing availability flag
        - waitingUpdate (bool): Pending update flag during processing
    """

    def __init__(self, parent):
        """
        Initialize request compute coordinator.
        
        Args:
            parent (EditImageModel): Parent model for result delivery
        """
        self.parent = parent

        self.requestDict= {} # store resqustCompute key:processNodeId, value: processNode params

        self.pool = QThreadPool.globalInstance()        # get global pool
        self.processpipe = None                         # processpipe ref

        self.readyToRun = True
        self.waitingUpdate = False

    def setProcessPipe(self,pp):
        """
        Set the current active processing pipeline.

        Args:
            pp (ProcessPipe, Required): Processing pipeline to manage
        """
        self.processpipe = pp
    
    def requestCompute(self, id, params):
        """
        Queue parameter update and request processing pipeline computation.
        
        Stores new parameters for a process-node and triggers pipeline
        computation. If processing is already running, marks for restart
        when current computation completes.

        Args:
            id (int, Required): Index of process-node in processing pipeline
            params (dict, Required): Updated parameters for process-node
        """
        self.requestDict[id] = copy.deepcopy(params)

        if self.readyToRun:
            # start processing processpipe
            self.pool.start(RunCompute(self))
        else:
            # if a computation is already running
            self.waitingUpdate = True

    def endCompute(self):
        """
        Handle processing completion and manage restart.
        
        Called when process-node computation finishes. Retrieves processed
        image, sends it to parent model, and restarts computation if there
        are pending parameter updates.
        """
        imgTM = self.processpipe.getImage(toneMap=True)
        self.parent.updateImage(imgTM)
        if self.waitingUpdate:
            self.pool.start(RunCompute(self))
            self.waitingUpdate = False
# -----------------------------------------------------------------------------
# --- Class RunCompute --------------------------------------------------------
# -----------------------------------------------------------------------------
class RunCompute(QRunnable):
    """
    Worker thread for HDR processing pipeline execution.
    
    Executes ProcessPipe computation on a dedicated thread, supporting both
    Python and C++ acceleration modes. Handles parameter updates and manages
    processing completion callbacks.
    
    Attributes:
        - parent (RequestCompute): Parent coordinator for completion callback
    """
    def __init__(self,parent):
        """
        Initialize compute worker thread.
        
        Args:
            parent (RequestCompute): Parent coordinator instance
        """
        super().__init__()
        self.parent = parent

    def run(self):
        """
        Main thread execution method.
        
        Executes ProcessPipe computation using either C++ acceleration
        or Python processing. Updates pipeline parameters, performs
        computation, and triggers completion callback.
        """
        self.parent.readyToRun = False
        for k in self.parent.requestDict.keys(): self.parent.processpipe.setParameters(k,self.parent.requestDict[k])
        cpp = True
        if cpp:
            img  = copy.deepcopy(self.parent.processpipe.getInputImage())
            imgRes = hdrCore.coreC.coreCcompute(img, self.parent.processpipe)
            self.parent.processpipe.setOutput(imgRes)
            self.parent.readyToRun = True
            self.parent.endCompute()
        else:
            start = timer()
            self.parent.processpipe.compute()
            dt = timer() - start
            self.parent.readyToRun = True
            self.parent.endCompute()
# -----------------------------------------------------------------------------
# --- Class RequestLoadImage --------------------------------------------------
# -----------------------------------------------------------------------------
class RequestLoadImage(object):
    """
    Parallel image loading coordinator for gallery thumbnail generation.
    
    Manages multi-threaded loading of images for gallery display, creating
    new threads for each image load operation. Handles load completion
    callbacks and error recovery with automatic retry mechanisms.
    
    Attributes:
        - parent (ImageGalleryModel): Parent model for image registration
        - pool (QThreadPool): Qt thread pool for worker management
        - requestsDone (dict): Load completion tracking {imageIndex: completed}
    """

    def __init__(self, parent):
        """
        Initialize image loading coordinator.
        
        Args:
            parent (ImageGalleryModel): Parent gallery model
        """
        self.parent = parent
        self.pool = QThreadPool.globalInstance()        # get a global pool
        self.requestsDone = {}

    def requestLoad(self, minIdxInPage, imgIdxInPage, filename):
        """
        Start image loading for specified gallery position.
        
        Args:
            minIdxInPage (int, Required): First image index in current page
            imgIdxInPage (int, Required): Relative image index within page
            filename (str, Required): Image file path to load
        """
        self.requestsDone[minIdxInPage+ imgIdxInPage] = False
        self.pool.start(RunLoadImage(self,minIdxInPage, imgIdxInPage,filename))

    def endLoadImage(self,error,idx0, idx,processPipe, filename):
        """
        Handle image loading completion or error recovery.
        
        Called when loading completes or fails (IOError, ValueError).
        Updates parent model with ProcessPipe and refreshes view on success.
        Automatically retries loading on failure.

        Args:
            error (bool, Required): True if loading failed
            idx0 (int, Required): First image index in current page
            idx (int, Required): Relative image index within page
            processPipe (ProcessPipe, Required): Loaded image processing pipeline
            filename (str, Required): Image filename
        """
        if not error:
            self.requestsDone[idx0 + idx] = True
            self.parent.processPipes[idx0 + idx]= processPipe
            self.parent.controller.view.updateImage(idx,processPipe, filename)
        else:
            self.requestLoad(idx0, idx, filename)
# -----------------------------------------------------------------------------
# --- Class RunLoadImage ------------------------------------------------------
# -----------------------------------------------------------------------------
class RunLoadImage(QRunnable):
    """
    Worker thread for individual image loading and ProcessPipe creation.
    
    Loads HDR images, creates associated ProcessPipes with default parameters,
    and performs initial computation for thumbnail generation. Handles
    loading errors gracefully with error reporting.
    
    Attributes:
        - parent (RequestLoadImage): Parent coordinator for completion callback
        - minIdxInPage (int): First image index in current page
        - imgIdxInPage (int): Relative image index within page
        - filename (str): Image file path to load
    """
    def __init__(self,parent, minIdxInPage, imgIdxInPage, filename):
        """
        Initialize image loading worker thread.
        
        Args:
            parent (RequestLoadImage): Parent coordinator instance
            minIdxInPage (int): First image index in current page
            imgIdxInPage (int): Relative image index within page
            filename (str): Image file path to load
        """
        super().__init__()
        self.parent = parent
        self.minIdxInPage = minIdxInPage
        self.imgIdxInPage = imgIdxInPage
        self.filename = filename

    def run(self):
        """
        Main thread execution for image loading.
        
        Loads image file, creates default ProcessPipe, performs initial
        computation, and reports completion or error to parent coordinator.
        Handles IOError and ValueError exceptions gracefully.
        """
        try:
            image_ = hdrCore.image.Image.read(self.filename, thumb=True)
            processPipe = model.EditImageModel.buildProcessPipe()
            processPipe.setImage(image_)                      
            processPipe.compute()
            self.parent.endLoadImage(False, self.minIdxInPage, self.imgIdxInPage, processPipe, self.filename)
        except(IOError, ValueError) as e:
            self.parent.endLoadImage(True, self.minIdxInPage, self.imgIdxInPage, None, self.filename)
# -----------------------------------------------------------------------------
# --- Class pCompute ----------------------------------------------------------
# -----------------------------------------------------------------------------
class pCompute(object):
    """
    Multi-threaded HDR processing with image splitting for large images.
    
    Manages parallel computation of ProcessPipe operations for HDR display
    or export by splitting large images into multiple parts and processing
    them concurrently. Provides progress feedback and handles geometry
    processing separately at the end for optimal performance.
    
    Processing Workflow:
    1. Split input image into grid of smaller sections
    2. Create duplicate ProcessPipes for each split
    3. Process splits in parallel using thread pool
    4. Merge processed splits back into complete image
    5. Apply geometry processing to final merged result
    6. Callback with final processed image
    
    Attributes:
        - callBack (function): Completion callback function
        - progress (function): Progress update callback function
        - nbSplits (int): Total number of image splits to process
        - nbDone (int): Number of completed split processing operations
        - geometryNode (ProcessNode): Geometry processing node for final step
        - meta (metadata): Image metadata for preservation
    """

    def __init__(self, callBack, processpipe,nbWidth,nbHeight, toneMap=True, progress=None, meta=None):
        """
        Initialize parallel HDR processing with image splitting.
        
        Args:
            callBack (function): Function called when processing completes
            processpipe (ProcessPipe): HDR processing pipeline to apply
            nbWidth (int): Number of horizontal splits
            nbHeight (int): Number of vertical splits  
            toneMap (bool): Apply tone mapping to final result
            progress (function, optional): Progress callback function
            meta (metadata, optional): Image metadata to preserve
        """
        self.callBack = callBack
        self.progress =progress
        self.nbSplits = nbWidth*nbHeight
        self.nbDone = 0
        self.geometryNode = None
        self.meta = meta
        # recover and split image
        input =  processpipe.getInputImage()

        # store last processNode (geometry) and remove it from processpipe
        if isinstance(processpipe.processNodes[-1].process,hdrCore.processing.geometry):
            self.geometryNode = copy.deepcopy(processpipe.processNodes[-1]) 

            # remove geometry node (the last one) 
            processpipe.processNodes = processpipe.processNodes[:-1]
       
        # split image and store splited images
        self.splits = input.split(nbWidth,nbHeight)

        self.pool = QThreadPool.globalInstance() 

        # duplicate processpipe, set image split and start
        for idxY,line in enumerate(self.splits):
            for idxX,split in enumerate(line):
                pp = copy.deepcopy(processpipe)
                pp.setImage(split)
                # start compute
                self.pool.start(pRun(self,pp,toneMap,idxX,idxY))

    def endCompute(self,idx,idy, split):
        """
        Handle completion of individual split processing.
        
        Called when each split finishes processing. Tracks completion progress,
        updates progress callback, and triggers final merge and geometry
        processing when all splits are complete.
        
        Args:
            idx (int): X-coordinate of completed split
            idy (int): Y-coordinate of completed split
            split (Image): Processed image split result
        """
        self.splits[idy][idx]= copy.deepcopy(split)
        self.nbDone += 1
        if self.progress:
            percent = str(int(self.nbDone*100/self.nbSplits))+'%'
            self.progress('HDR image process-pipe computation:'+percent)
        if self.nbDone == self.nbSplits:
            res = hdrCore.image.Image.merge(self.splits)
            # process geometry
            if self.geometryNode:
                res = self.geometryNode.process.compute(res,**self.geometryNode.params)
            # callBack caller
            self.callBack(res, self.meta)
# -----------------------------------------------------------------------------
# --- Class pRun --------------------------------------------------------------
# -----------------------------------------------------------------------------
class pRun(QRunnable):
    """
    Worker thread for processing individual image splits in parallel.
    
    Executes ProcessPipe computation on a single image split, supporting
    both tone-mapped and HDR output modes. Reports completion back to
    parent pCompute coordinator.
    
    Attributes:
        - parent (pCompute): Parent coordinator for completion callback
        - processpipe (ProcessPipe): Processing pipeline to apply to split
        - idx (tuple): (x, y) coordinates of split in grid
        - toneMap (bool): Apply tone mapping to result
    """
    def __init__(self,parent,processpipe,toneMap, idxX,idxY):
        """
        Initialize split processing worker thread.
        
        Args:
            parent (pCompute): Parent coordinator instance
            processpipe (ProcessPipe): Processing pipeline to apply
            toneMap (bool): Apply tone mapping to result
            idxX (int): X-coordinate of split in grid
            idxY (int): Y-coordinate of split in grid
        """
        super().__init__()
        self.parent = parent
        self.processpipe = processpipe
        self.idx = (idxX,idxY)
        self.toneMap = toneMap

    def run(self):
        """
        Main thread execution for split processing.
        
        Executes ProcessPipe computation on the assigned image split
        and reports completion with processed result to parent coordinator.
        """
        self.processpipe.compute()
        pRes = self.processpipe.getImage(toneMap=self.toneMap)
        self.parent.endCompute(self.idx[0],self.idx[1], pRes)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Class cCompute ----------------------------------------------------------
# -----------------------------------------------------------------------------
class cCompute(object):
    """
    C++ accelerated HDR processing for high-performance computation.
    
    Provides hardware-accelerated HDR processing using C++ backend
    for improved performance on large images or complex operations.
    Uses single-threaded execution with optimized C++ implementation.
    
    Attributes:
        - callBack (function): Completion callback function
        - progress (function): Progress update callback function
        - pool (QThreadPool): Qt thread pool for worker management
    """

    def __init__(self, callBack, processpipe, toneMap=True, progress=None):
        """
        Initialize C++ accelerated HDR processing.
        
        Args:
            callBack (function): Function called when processing completes
            processpipe (ProcessPipe): HDR processing pipeline to execute
            toneMap (bool): Apply tone mapping to result
            progress (function, optional): Progress callback function
        """
        self.callBack = callBack
        self.progress =progress

        # recover image
        input =  processpipe.getInputImage()

        self.pool = QThreadPool.globalInstance() 
        self.pool.start(cRun(self,processpipe,toneMap))

    def endCompute(self, img):
        """
        Handle processing completion and deliver result.
        
        Args:
            img (Image): Processed HDR image result
        """
        self.callBack(img)
# -----------------------------------------------------------------------------
# --- Class cRun --------------------------------------------------------------
# -----------------------------------------------------------------------------
class cRun(QRunnable):
    """
    Worker thread for C++ accelerated HDR processing execution.
    
    Executes ProcessPipe computation using C++ backend acceleration
    for improved performance. Handles input image copying and result
    delivery to parent coordinator.
    
    Attributes:
        - parent (cCompute): Parent coordinator for completion callback
        - processpipe (ProcessPipe): Processing pipeline to execute
        - toneMap (bool): Apply tone mapping to result
    """
    def __init__(self,parent,processpipe,toneMap):
        """
        Initialize C++ processing worker thread.
        
        Args:
            parent (cCompute): Parent coordinator instance
            processpipe (ProcessPipe): Processing pipeline to execute
            toneMap (bool): Apply tone mapping to result
        """
        super().__init__()
        self.parent = parent
        self.processpipe = processpipe
        self.toneMap = toneMap

    def run(self):
        """
        Main thread execution with C++ acceleration.
        
        Executes ProcessPipe computation using C++ backend (hdrCore.coreC)
        for improved performance. Copies input image, processes through
        C++ pipeline, and reports completion to parent coordinator.
        """
        img  = copy.deepcopy(self.processpipe.getInputImage())
        imgRes = hdrCore.coreC.coreCcompute(img, self.processpipe)
        self.processpipe.setOutput(imgRes)

        pRes = self.processpipe.getImage(toneMap=self.toneMap)
        self.parent.endCompute(pRes)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Class RequestAestheticsCompute ----------------------------------------------------
# -----------------------------------------------------------------------------
class RequestAestheticsCompute(object):
    """
    Threading coordinator for image aesthetics analysis and computation.
    
    Manages parallel computation of aesthetic analysis including color palette
    extraction, composition analysis, and visual quality assessment. Uses
    single-threaded sequential processing to maintain analysis order while
    providing non-blocking background computation.
    
    Similar to RequestCompute but specialized for aesthetics analysis workflows
    that may have different performance characteristics and requirements.
    
    Attributes:
        - parent: Parent model for result callbacks  
        - requestDict (dict): Stores pending parameter updates
        - pool (QThreadPool): Qt thread pool for worker management
        - processpipe (ProcessPipe): Active processing pipeline reference
        - readyToRun (bool): Processing availability flag
        - waitingUpdate (bool): Pending update flag during processing
    """

    def __init__(self, parent):
        """
        Initialize aesthetics compute coordinator.
        
        Args:
            parent: Parent model for result delivery
        """
        self.parent = parent

        self.requestDict= {} # store resqustCompute key:processNodeId, value: processNode params

        self.pool = QThreadPool.globalInstance()        # get global pool
        self.processpipe = None                         # processpipe ref

        self.readyToRun = True
        self.waitingUpdate = False

    def setProcessPipe(self,pp):
        """
        Set the current active processing pipeline.

        Args:
            pp (ProcessPipe, Required): Processing pipeline to manage
        """
        self.processpipe = pp
    
    def requestCompute(self, id, params):
        """
        Queue parameter update and request aesthetics computation.

        Args:
            id (int, Required): Index of process-node in processing pipeline
            params (dict, Required): Updated parameters for process-node
        """
        self.requestDict[id] = copy.deepcopy(params)

        if self.readyToRun:
            # start processing processpipe
            self.pool.start(RunAestheticsCompute(self))
        else:
            # if a computation is already running
            self.waitingUpdate = True

    def endCompute(self):
        """
        Handle aesthetics processing completion and manage restart.
        
        Called when aesthetics computation finishes. Retrieves processed
        image, sends it to parent model, and restarts computation if there
        are pending parameter updates.
        """
        imgTM = self.processpipe.getImage(toneMap=True)
        self.parent.updateImage(imgTM)
        if self.waitingUpdate:
            self.pool.start(RunAestheticsCompute(self))
            self.waitingUpdate = False
# -----------------------------------------------------------------------------
# --- Class RunCompute --------------------------------------------------------
# -----------------------------------------------------------------------------
class RunAestheticsCompute(QRunnable):
    """
    Worker thread for aesthetics analysis and computation execution.
    
    Executes ProcessPipe computation focused on aesthetics analysis including
    color palette extraction, composition metrics, and visual quality assessment.
    Uses Python-based processing for flexibility in analysis algorithms.
    
    Attributes:
        - parent (RequestAestheticsCompute): Parent coordinator for completion callback
    """
    def __init__(self,parent):
        """
        Initialize aesthetics compute worker thread.
        
        Args:
            parent (RequestAestheticsCompute): Parent coordinator instance
        """
        super().__init__()
        self.parent = parent

    def run(self):
        """
        Main thread execution for aesthetics processing.
        
        Executes ProcessPipe computation with focus on aesthetics analysis.
        Updates pipeline parameters, performs Python-based computation
        with timing metrics, and triggers completion callback.
        """
        self.parent.readyToRun = False
        for k in self.parent.requestDict.keys(): self.parent.processpipe.setParameters(k,self.parent.requestDict[k])
        start = timer()
        self.parent.processpipe.compute()
        dt = timer() - start
        self.parent.readyToRun = True
        self.parent.endCompute()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
