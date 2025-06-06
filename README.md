# uHDRv6 - HDR image processing application

**uHDRv6** is a graphical application dedicated to the processing and analysis of HDR (High Dynamic Range) images running under Windows.

## 👥 Contributors
- Rémi COZOT
- Rémi SYNVAVE
- Jhing ZHANG
- Arnaud WISSOCQ
- Justin FONTAINE
- Thibaut DUFEUTREL

## Installation
You need
- Windows 10/11
- Python 3.12
- HDRImageViewer.exe *(optional)*

You need to install the libraries required to launch the application
```bash
pip install -r requirements.txt
```
To launch the application:
```bash
python uHDR.py
```
To generate the documentation:
```bash
pdoc uHDR.py guiQt hdrCore -o docs
```
or execute the generate-doc.bat file

## 🔎 Architecture
```bash
uHDRv6
 |--guiQT       # Graphical interface
 |--hdrCore     # HDR processing
 |--preferences # User configuration
 |--thumbnails  # Temporary HDR thumbnail
 |-compOrigFinal.json   # ...
 |-exiftool.exe         # Software to retrieve image metadata
 |-grey.json            # ...
 |-requirements.txt     # List of required libraries
 |-temp.json            # ...
 |-uHDR.py              # Main application file
```

## ✉️ Contact
Rémi COZOT - remi.cozot@univ-littoral.fr