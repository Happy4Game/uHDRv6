# hdrCore project 2020
# author: remi.cozot@univ-littoral.fr


"""
uHDR v6 - HDR Image Editing Software

This is the main entry point for the uHDR v6 application, a High Dynamic Range (HDR) 
image editing software with C++ core processing. The application provides a Qt-based 
graphical interface for professional HDR image manipulation and processing.

Features:
    - HDR image loading and processing
    - C++ accelerated core processing
    - PyQt5-based user interface
    - Advanced HDR tone mapping
    - Multiprocessing support

Author: remi.cozot@univ-littoral.fr
Copyright (C) 2021 Remi Cozot
License: GNU General Public License v3.0
"""


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDesktopWidget
import guiQt.controller
from multiprocessing import freeze_support

import sys
# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    Main entry point for the uHDR v6 application.
    
    Initializes the PyQt5 application, creates the main controller, and starts the
    event loop. Includes multiprocessing freeze support for Windows compatibility.
    """
    freeze_support()
    print("uHDRv6 (C++ core)")

    app = QApplication(sys.argv)

    mcQt = guiQt.controller.AppController(app)

    sys.exit(app.exec_())
# ------------------------------------------------------------------------------------------
