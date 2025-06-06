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
# --- Package preferences -----------------------------------------------------
# -----------------------------------------------------------------------------
"""
uHDR Preferences Management Module

This module manages global application preferences and configuration settings
for the uHDR HDR image editing software. It handles loading and saving user
preferences, HDR display configurations, computation modes, and file paths.

The preferences system supports multiple HDR display types with different
scaling factors and output formats. All preferences are persisted to JSON
files for session continuity.

Global Variables:
    computation (str): Computation backend ('python', 'numba', 'cuda')
    verbose (bool): Enable verbose logging output
    HDRdisplays (dict): Available HDR display configurations
    HDRdisplay (str): Currently selected HDR display
    maxWorking (int): Maximum image resolution for editing operations
    imagePath (str): Default directory path for image operations
    keepAllMeta (bool): Whether to preserve all metadata during processing

Functions:
    loadPref: Load preferences from JSON configuration file
    savePref: Save current preferences to JSON file
    getComputationMode: Get current computation backend setting
    getHDRdisplays: Get all available HDR display configurations
    getHDRdisplay: Get current HDR display configuration
    setHDRdisplay: Set the active HDR display type
    getDisplayScaling: Get scaling factor for current display
    getDisplayShape: Get resolution for current display
    getImagePath: Get current image directory path
    setImagePath: Set image directory path and save preferences
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
# RCZT 2023
# import numba, json, os, copy
import numpy as np, json, os

# -----------------------------------------------------------------------------
# --- Preferences -------------------------------------------------------------
# -----------------------------------------------------------------------------
target = ['python','numba','cuda']
computation = target[0]
# verbose mode: print function call 
#   usefull for debug
verbose = True
# list of HDR display takien into account
#   red from prefs.json file
#   display info:
#   "vesaDisplayHDR1000":                           << display tag name
#       {
#           "shape": [ 2160, 3840 ],                << display shape (4K)
#           "scaling": 12,                          << color space scaling to max
#           "post": "_vesa_DISPLAY_HDR_1000",       << postfix add when exporting file
#           "tag": "vesaDisplayHDR1000"             << tag name
#       }
HDRdisplays = None
# current HDR display: tag name in above list
HDRdisplay = None
# image size when editing image: 
#   small size = quick computation, no memory issues
maxWorking = 1200
# last image directory path
imagePath ="."
# keep all metadata
keepAllMeta = False
# -----------------------------------------------------------------------------
# --- Functions preferences --------------------------------------------------
# -----------------------------------------------------------------------------
def loadPref(): 
    """
    Load application preferences from the configuration file.
    
    Reads preferences from './preferences/prefs.json' and returns the
    configuration dictionary. This includes HDR display settings,
    current display selection, and image path preferences.

    Returns:
        dict or None: Preferences dictionary if file exists and is valid,
                     None if file cannot be read or parsed
                     
    Note:
        The preferences file should contain HDR display configurations,
        current display selection, and default image paths.
    """
    with open('./preferences/prefs.json') as f: return  json.load(f)
# -----------------------------------------------------------------------------
def savePref():
    """
    Save current preferences to the configuration file.
    
    Writes the current HDR display configurations, selected display,
    and image path to './preferences/prefs.json' for persistence
    across application sessions.
    
    The saved preferences include:
    - HDRdisplays: All available display configurations
    - HDRdisplay: Currently selected display tag
    - imagePath: Current default image directory
    """
    global HDRdisplays
    global HDRdisplay
    global imagePath
    pUpdate = {
            "HDRdisplays" : HDRdisplays,
            "HDRdisplay"  : HDRdisplay,
            "imagePath"   : imagePath
        }
    if verbose: print(" [PREF] >> savePref(",pUpdate,")")
    with open('./preferences/prefs.json', "w") as f: json.dump(pUpdate,f)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# loading pref
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
print("uHDRv6: loading preferences")
p = loadPref()
if p :
    HDRdisplays = p["HDRdisplays"]
    HDRdisplay = p["HDRdisplay"]
    imagePath = p["imagePath"]
else:
    HDRdisplays = {
        'none' :                {'shape':(2160,3840), 'scaling':1,   'post':'',                          'tag': "none"},
        'vesaDisplayHDR1000' :  {'shape':(2160,3840), 'scaling':12,  'post':'_vesa_DISPLAY_HDR_1000',    'tag':'vesaDisplayHDR1000'},
        'vesaDisplayHDR400' :   {'shape':(2160,3840), 'scaling':4.8, 'post':'_vesa_DISPLAY_HDR_400',     'tag':'vesaDisplayHDR400'},
        'HLG1' :                {'shape':(2160,3840), 'scaling':1,   'post':'_HLG_1',                    'tag':'HLG1'}
        }
    # current display
    HDRdisplay = 'vesaDisplayHDR1000'
    imagePath = '.'
print(f"       target display: {HDRdisplay}")
print(f"       image path: {imagePath}")
# -----------------------------------------------------------------------------
# --- Functions computation ---------------------------------------------------
# -----------------------------------------------------------------------------
def getComputationMode():
    """
    Get the current computation backend mode.
    
    Returns the currently configured computation backend that determines
    which processing implementation to use for image operations.

    Returns:
        str: Current computation mode ('python', 'numba', or 'cuda')
        
    Note:
        - 'python': Standard Python/NumPy implementation
        - 'numba': Numba JIT-compiled acceleration
        - 'cuda': GPU acceleration via CUDA
    """
    return computation
# -----------------------------------------------------------------------------
# --- Functions HDR dispaly ---------------------------------------------------
# -----------------------------------------------------------------------------
def getHDRdisplays():
    """
    Get all available HDR display configurations.
    
    Returns the complete dictionary of HDR display configurations including
    their properties such as resolution, scaling factors, and export settings.

    Returns:
        dict: Dictionary of HDR display configurations where keys are display
              names and values contain display properties:
              - shape: Display resolution (height, width)
              - scaling: HDR scaling factor
              - post: Filename postfix for exports
              - tag: Display identifier tag
    """
    return HDRdisplays
# -----------------------------------------------------------------------------
def getHDRdisplay():
    """
    Get the current HDR display configuration.
    
    Returns the configuration dictionary for the currently selected HDR
    display, containing all properties needed for HDR output.

    Returns:
        dict: Current HDR display configuration containing:
              - shape: Display resolution tuple (height, width)
              - scaling: HDR luminance scaling factor
              - post: Filename postfix for exports
              - tag: Display identifier string
    """
    return HDRdisplays[HDRdisplay]
# -----------------------------------------------------------------------------
def setHDRdisplay(tag):
    """
    Set the active HDR display type.
    
    Changes the current HDR display configuration to the specified display
    tag and saves the preference to persistent storage.

    Args:
        tag (str): HDR display tag that must exist in HDRdisplays dictionary
                   (e.g., 'vesaDisplayHDR1000', 'vesaDisplayHDR400', 'none')
                   
    Note:
        The display tag must be a valid key in the HDRdisplays dictionary.
        Preferences are automatically saved after a successful change.
    """
    global HDRdisplay
    if tag in HDRdisplays: HDRdisplay =tag
    savePref()
# ----------------------------------------------------------------------------
def getDisplayScaling():  
    """
    Get the HDR scaling factor for the current display.
    
    Returns the luminance scaling factor used to convert HDR values
    for the currently selected display type.
    
    Returns:
        float: HDR scaling factor (e.g., 12.0 for HDR1000, 4.8 for HDR400)
    """
    return getHDRdisplay()['scaling']
# ----------------------------------------------------------------------------
def getDisplayShape():  
    """
    Get the resolution of the current HDR display.
    
    Returns the display resolution as a tuple for the currently
    selected HDR display configuration.
    
    Returns:
        tuple: Display resolution as (height, width) in pixels
    """
    return getHDRdisplay()['shape']
# -----------------------------------------------------------------------------
# --- Functions path ---------------------------------------------------
# -----------------------------------------------------------------------------
def getImagePath(): 
    """
    Get the current default image directory path.
    
    Returns the configured image directory path, with fallback to current
    directory if the configured path doesn't exist.
    
    Returns:
        str: Valid directory path for image operations. Returns '.' if the
             configured imagePath doesn't exist or is invalid.
    """
    return imagePath if os.path.isdir(imagePath) else '.'
# ----------------------------------------------------------------------------
def setImagePath(path): 
    """
    Set the default image directory path and save preferences.
    
    Updates the global image path setting and persists the change to the
    preferences file for use in future sessions.
    
    Args:
        path (str): Directory path to set as the default image location
        
    Note:
        The path is not validated before setting. Preferences are automatically
        saved after the path is updated.
    """
    global imagePath
    imagePath = path
    if verbose: print(" [PREF] >> setImagePath(",path,"):",imagePath)
    savePref()
# ----------------------------------------------------------------------------
             