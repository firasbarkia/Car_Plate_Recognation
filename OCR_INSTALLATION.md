# OCR Installation Guide

This application supports multiple OCR engines for license plate recognition. You can install one or more of them.

## Recommended: PaddleOCR (Best for Arabic/Tunisian Plates)

```bash
pip install paddlepaddle paddleocr
```

**Note:** PaddleOCR is highly recommended for Arabic/Tunisian license plates as it has excellent Arabic character recognition.

## Alternative 1: EasyOCR

```bash
pip install easyocr
```

## Alternative 2: Tesseract OCR

**Important:** Tesseract requires both the Python package AND the Tesseract binary.

### Windows:
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it (default path: `C:\Program Files\Tesseract-OCR`)
3. Install Python package:
```bash
pip install pytesseract
```

### Linux:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ara
pip install pytesseract
```

### macOS:
```bash
brew install tesseract tesseract-lang
pip install pytesseract
```

## Install All (Recommended)

To get the best results, install all three:

```bash
pip install easyocr paddlepaddle paddleocr pytesseract
```

Then install Tesseract binary separately (see above for your OS).

## Usage in App

- **PaddleOCR**: Best for Arabic/Tunisian plates, fast, accurate
- **EasyOCR**: Good general purpose, supports many languages
- **Tesseract**: Classic OCR, requires binary installation

You can select which engine to use in the app sidebar, or enable "Try All Available Engines" to compare results automatically.

