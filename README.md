# uHDRv6 - HDR image processing application

**uHDRv6** is a graphical application dedicated to the processing and analysis of HDR (High Dynamic Range) images running under Windows.

## 👥 Contributors
- Rémi COZOT
- Rémi SYNVAVE
- ..
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