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
HDR Core C++ Integration Module

This module provides high-performance C++ integration for HDR image processing
through dynamic library loading. It interfaces with the HDRip.dll C++ library
to accelerate computationally intensive image processing operations.

The module implements a fixed processing pipeline architecture optimized for
speed, including exposure adjustment, contrast control, tone curves, lightness
masking, saturation adjustment, and multiple color editors.

Functions:
    coreCcompute: Main C++ processing pipeline execution function
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import ctypes, copy
import numpy as np
import hdrCore.image, hdrCore.processing, hdrCore.utils
import preferences.preferences as pref

# -----------------------------------------------------------------------------
# --- coreCcompute ------------------------------------------------------------
# -----------------------------------------------------------------------------

def coreCcompute(img, processPipe):
    """
    Execute the complete HDR processing pipeline using C++ acceleration.
    
    This function provides high-speed image processing by interfacing with a
    C++ dynamic library (HDRip.dll). It implements a fixed pipeline architecture
    consisting of 10 processing stages: exposure, contrast, tone curve, lightness
    mask, saturation, and 5 color editors.
    
    The processing pipeline is optimized for performance and follows this sequence:
    1. Exposure adjustment
    2. Contrast control  
    3. Tone curve mapping (shadows, blacks, mediums, whites, highlights)
    4. Lightness mask application
    5. Saturation adjustment
    6-10. Five independent color editors with selection and editing capabilities
    
    Args:
        img (hdrCore.image.Image): Input HDR image to be processed
        processPipe (hdrCore.processing.ProcessPipe): Processing pipeline containing
                                                     all processing parameters in the
                                                     expected fixed architecture
            
    Returns:
        hdrCore.image.Image: Processed image with C++ accelerated operations applied
        
    Note:
        This function requires HDRip.dll to be present in the current directory.
        The processing pipeline architecture is fixed and cannot be modified.
        Each color editor supports selection by lightness, chroma, and hue ranges
        with tolerance-based masking and independent edit controls for hue,
        exposure, contrast, and saturation.
        
    Raises:
        OSError: If HDRip.dll cannot be loaded
        ctypes.ArgumentError: If parameter types don't match expected C++ interface
    """
    if pref.verbose:  print(f"[hdrCore] >> coreCcompute({img})") 

    ppDict = processPipe.toDict()

    exposure = ppDict[0]['exposure']['EV']

    contrast = ppDict[1]['contrast']['contrast']

    tonecurveS = ppDict[2]['tonecurve']['shadows'][1]
    tonecurveB = ppDict[2]['tonecurve']['blacks'][1]
    tonecurveM = ppDict[2]['tonecurve']['mediums'][1]
    tonecurveW = ppDict[2]['tonecurve']['whites'][1]
    tonecurveH = ppDict[2]['tonecurve']['highlights'][1]

    lightnessMaskS = ppDict[3]['lightnessmask']['shadows']
    lightnessMaskB = ppDict[3]['lightnessmask']['blacks']
    lightnessMaskM = ppDict[3]['lightnessmask']['mediums']
    lightnessMaskW = ppDict[3]['lightnessmask']['whites']
    lightnessMaskH = ppDict[3]['lightnessmask']['highlights']

    saturation = ppDict[4]['saturation']['saturation']

    ce1_sel_lightness = ppDict[5]['colorEditor0']['selection']['lightness']
    ce1_sel_chroma = ppDict[5]['colorEditor0']['selection']['chroma']
    ce1_sel_hue = ppDict[5]['colorEditor0']['selection']['hue'] 
    ce1_tolerance = 0.1
    ce1_edit_hue = ppDict[5]['colorEditor0']['edit']['hue'] if 'hue' in ppDict[5]['colorEditor0']['edit'] else 0.0
    ce1_edit_exposure = ppDict[5]['colorEditor0']['edit']['exposure']
    ce1_edit_contrast = ppDict[5]['colorEditor0']['edit']['contrast']
    ce1_edit_saturation = ppDict[5]['colorEditor0']['edit']['saturation']
    ce1_mask = ppDict[5]['colorEditor0']['mask']

    ce2_sel_lightness = ppDict[6]['colorEditor1']['selection']['lightness']
    ce2_sel_chroma = ppDict[6]['colorEditor1']['selection']['chroma']
    ce2_sel_hue = ppDict[6]['colorEditor1']['selection']['hue']
    ce2_tolerance = 0.1
    ce2_edit_hue = ppDict[6]['colorEditor1']['edit']['hue'] if 'hue' in ppDict[6]['colorEditor1']['edit'] else 0.0
    ce2_edit_exposure = ppDict[6]['colorEditor1']['edit']['exposure']
    ce2_edit_contrast = ppDict[6]['colorEditor1']['edit']['contrast']
    ce2_edit_saturation = ppDict[6]['colorEditor1']['edit']['saturation']
    ce2_mask = ppDict[6]['colorEditor1']['mask']

    ce3_sel_lightness = ppDict[7]['colorEditor2']['selection']['lightness']
    ce3_sel_chroma = ppDict[7]['colorEditor2']['selection']['chroma']
    ce3_sel_hue = ppDict[7]['colorEditor2']['selection']['hue']
    ce3_tolerance = 0.1
    ce3_edit_hue = ppDict[7]['colorEditor2']['edit']['hue'] if 'hue' in ppDict[7]['colorEditor2']['edit'] else 0.0
    ce3_edit_exposure = ppDict[7]['colorEditor2']['edit']['exposure']
    ce3_edit_contrast = ppDict[7]['colorEditor2']['edit']['contrast']
    ce3_edit_saturation = ppDict[7]['colorEditor2']['edit']['saturation']
    ce3_mask = ppDict[7]['colorEditor2']['mask']

    ce4_sel_lightness = ppDict[8]['colorEditor3']['selection']['lightness']
    ce4_sel_chroma = ppDict[8]['colorEditor3']['selection']['chroma']
    ce4_sel_hue = ppDict[8]['colorEditor3']['selection']['hue']
    ce4_tolerance = 0.1
    ce4_edit_hue = ppDict[8]['colorEditor3']['edit']['hue'] if 'hue' in ppDict[8]['colorEditor3']['edit'] else 0.0
    ce4_edit_exposure = ppDict[8]['colorEditor3']['edit']['exposure']
    ce4_edit_contrast = ppDict[8]['colorEditor3']['edit']['contrast']
    ce4_edit_saturation = ppDict[8]['colorEditor3']['edit']['saturation']
    ce4_mask = ppDict[8]['colorEditor3']['mask']

    ce5_sel_lightness = ppDict[9]['colorEditor4']['selection']['lightness']
    ce5_sel_chroma = ppDict[9]['colorEditor4']['selection']['chroma']
    ce5_sel_hue = ppDict[9]['colorEditor4']['selection']['hue']
    ce5_tolerance = 0.1
    ce5_edit_hue = ppDict[9]['colorEditor4']['edit']['hue'] if 'hue' in ppDict[9]['colorEditor4']['edit'] else 0.0
    ce5_edit_exposure = ppDict[9]['colorEditor4']['edit']['exposure']
    ce5_edit_contrast = ppDict[9]['colorEditor4']['edit']['contrast']
    ce5_edit_saturation = ppDict[9]['colorEditor4']['edit']['saturation']
    ce5_mask = ppDict[9]['colorEditor4']['mask']



    mylib = ctypes.cdll.LoadLibrary('./HDRip.dll')
    mylib.full_process_5CO.argtypes = [np.ctypeslib.ndpointer(dtype=ctypes.c_float), ctypes.c_uint, ctypes.c_uint,
                                    ctypes.c_float,
                                    ctypes.c_float,
                                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                    ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_float,
                                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool
    ]
    mylib.full_process_5CO.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_float, shape=(img.colorData.shape[0],img.colorData.shape[1],3))

    resDLL = mylib.full_process_5CO(img.colorData,
                                img.colorData.shape[1],
                                img.colorData.shape[0],
                                exposure,
                                contrast,
                                tonecurveS, tonecurveB, tonecurveM, tonecurveW, tonecurveH,
                                lightnessMaskS, lightnessMaskB, lightnessMaskM, lightnessMaskW, lightnessMaskH,
                                saturation,
                                ce1_sel_lightness[0], ce1_sel_lightness[1], ce1_sel_chroma[0], ce1_sel_chroma[1], ce1_sel_hue[0], ce1_sel_hue[1], ce1_tolerance, ce1_edit_hue, ce1_edit_exposure, ce1_edit_contrast, ce1_edit_saturation, ce1_mask,
                                ce2_sel_lightness[0], ce2_sel_lightness[1], ce2_sel_chroma[0], ce2_sel_chroma[1], ce2_sel_hue[0], ce2_sel_hue[1], ce2_tolerance, ce2_edit_hue, ce2_edit_exposure, ce2_edit_contrast, ce2_edit_saturation, ce2_mask,
                                ce3_sel_lightness[0], ce3_sel_lightness[1], ce3_sel_chroma[0], ce3_sel_chroma[1], ce3_sel_hue[0], ce3_sel_hue[1], ce3_tolerance, ce3_edit_hue, ce3_edit_exposure, ce3_edit_contrast, ce3_edit_saturation, ce3_mask,
                                ce4_sel_lightness[0], ce4_sel_lightness[1], ce4_sel_chroma[0], ce4_sel_chroma[1], ce4_sel_hue[0], ce4_sel_hue[1], ce4_tolerance, ce4_edit_hue, ce4_edit_exposure, ce4_edit_contrast, ce4_edit_saturation, ce4_mask,
                                ce5_sel_lightness[0], ce5_sel_lightness[1], ce5_sel_chroma[0], ce5_sel_chroma[1], ce5_sel_hue[0], ce5_sel_hue[1], ce5_tolerance, ce5_edit_hue, ce5_edit_exposure, ce5_edit_contrast, ce5_edit_saturation, ce5_mask
                                )

    img.colorData = copy.deepcopy(resDLL)

    return img
