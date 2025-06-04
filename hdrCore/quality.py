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
HDR Core Quality Assessment Module

This module provides functionality for assessing and recording image quality
metrics and artifacts. It includes user-based quality scoring systems and
artifact detection capabilities for HDR image evaluation.

Classes:
    quality: Image quality assessment and artifact tracking system
"""





# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import json, os, copy
import numpy as np
from . import utils, processing, image
# -----------------------------------------------------------------------------
# --- Class quality ----------------------------------------------------------
# -----------------------------------------------------------------------------
class quality(object):
    """
    Image quality assessment and artifact tracking system.
    
    This class provides a comprehensive framework for evaluating image quality
    through user scoring and artifact detection. It tracks multiple quality
    dimensions including aesthetics, comfort, and naturalness, along with
    common HDR processing artifacts.
    
    Attributes:
        _image (hdrCore.image.Image): Reference to the assessed image
        imageNpath (dict): Image identification containing name and path
        user (dict): User information for the assessment
        score (dict): Quality scores across multiple dimensions:
            - quality: Overall quality score
            - aesthetics: Aesthetic appeal score  
            - comfort: Visual comfort score
            - naturalness: Naturalness/realism score
        artifact (dict): Boolean flags for detected artifacts:
            - ghost: Ghosting artifacts
            - noise: Noise artifacts
            - blur: Blur artifacts
            - halo: Halo artifacts
            - other: Other unspecified artifacts
    """
    
    def __init__(self):
        """
        Initialize a new quality assessment object.
        
        Creates a new quality assessment with default values for all
        scoring dimensions and artifact flags. All scores are initialized
        to 0 and all artifact flags to False.
        """
        self._image =       None
        self.imageNpath =    {'name':None, 'path': None}
        self.user =         {'pseudo': None}
        self.score =        {'quality': 0,'aesthetics':0, 'confort':0,'naturalness':0}
        self.artifact =     {'ghost':False, 'noise':False, 'blur':False, 'halo':False, 'other':False}

    def toDict(self):
        """
        Convert the quality assessment to a dictionary representation.
        
        Creates a dictionary containing all assessment data including image
        information, user details, scores, and artifact flags. This format
        is suitable for serialization and storage.
        
        Returns:
            dict: Complete quality assessment data with keys:
                - image: Image name and path information
                - user: User identification data
                - score: All quality dimension scores
                - artifact: All artifact detection flags
        """
        return {'image': copy.deepcopy(self.imageNpath),
                              'user':copy.deepcopy(self.user),
                              'score':copy.deepcopy(self.score),
                              'artifact':copy.deepcopy(self.artifact)}

    def __repr__(self):
        """
        Return a string representation of the quality assessment.
        
        Returns:
            str: String representation of the complete assessment data
        """
        return str(self.toDict())

    def __str__(self):
        """
        Return a human-readable string representation of the quality assessment.
        
        Returns:
            str: Human-readable string of the complete assessment data
        """
        return self.__repr__()
