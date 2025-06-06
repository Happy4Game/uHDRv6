# uHDRv6 - Application de traitement d'image HDR

**uHDRv6** est une application graphique dédiée au traitement et à l'analyse des images HDR (High Dynamic Range) fonctionnant sous Windows.

## 👥 Contributeurs
- Rémi COZOT
- Rémi SYNVAVE
- Jing ZHANG
- Arnaud WISSOCQ
- Justin FONTAINE
- Thibaut DUFEUTREL

## Installation
Il vous faut impérativement
- Windows 10/11
- Python 3.12
- HDRImageViewer.exe *(optionnel)*

Il faut installer les bibliothèques nécessaires pour lancer l'application
```bash
pip install -r requirements.txt
```
Pour lancer l'application:
```bash
python uHDR.py
```

## 🔎 Architecture
```bash
uHDRv6
 |--guiQT       # Interface graphique 
 |--hdrCore     # Traitement HDR
 |--preferences # Configuration utilisateur
 |--thumbnails  # Mignature HDR temporaire
 |-compOrigFinal.json   # ...
 |-exiftool.exe         # Logiciel permettant de récupérer les métadonnées des images
 |-grey.json            # ...
 |-requirements.txt     # Liste de bibliothèques requises
 |-temp.json            # ...
 |-uHDR.py              # Fichier principal de l'application
```

## ✉️ Contact
Rémi COZOT - remi.cozot@univ-littoral.fr