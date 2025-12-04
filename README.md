# 🚗 Car Detection & License Plate Recognition System

A comprehensive computer vision system for detecting vehicles and recognizing license plates using YOLO object detection and multiple OCR engines. The system supports Arabic and English characters, making it suitable for Tunisian license plates.

## ✨ Features

### 🔍 Complete Detection Pipeline

1. **Image Quality Enhancement**
   - Noise reduction using fastNlMeansDenoisingColored
   - Contrast correction with CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Brightness and luminance adjustment
   - Configurable enhancement parameters

2. **Vehicle Detection**
   - YOLO-based vehicle detection
   - Classification (car, truck, motorcycle, etc.)
   - Bounding box visualization
   - Region extraction for further processing

3. **License Plate Detection**
   - Two-stage pipeline: vehicles first, then plates
   - YOLO model specifically trained for license plate detection
   - Precise localization with bounding boxes
   - Detection on extracted vehicle regions

4. **License Plate Preprocessing**
   - Automatic resizing and scaling
   - Multiple preprocessing methods:
     - OTSU Threshold
     - Adaptive Threshold
     - Morphological operations
     - Gaussian Blur + OTSU
     - CLAHE + OTSU
   - Contrast enhancement
   - Noise removal

5. **OCR Recognition**
   - Support for 3 OCR engines:
     - **EasyOCR** - Fast and accurate
     - **PaddleOCR** - Recommended for Arabic/Tunisian plates
     - **Tesseract** - Open-source alternative
   - Multilingual support (Arabic + English)
   - Automatic best result selection
   - Comparison across multiple engines

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CarDetection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install OCR engines (optional but recommended)**
   
   For **PaddleOCR** (recommended for Arabic):
   ```bash
   pip install paddlepaddle paddleocr
   ```
   
   For **EasyOCR**:
   ```bash
   pip install easyocr
   ```
   
   For **Tesseract**:
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`
   - Then: `pip install pytesseract`
   
   See [OCR_INSTALLATION.md](OCR_INSTALLATION.md) for detailed instructions.

5. **Download models**
   
   Ensure you have the following model files:
   - `car_detection_model.pt` - Vehicle detection model
   - `runs/detect/plate_detection_yolo11/weights/best.pt` - License plate detection model
   
   If models are missing, train them using the provided Jupyter notebooks:
   - `train_car_detection_model.ipynb`
   - `train_plate_recognition_model.ipynb`

### Running the Application

#### Main Application (Complete Pipeline)

Run the complete pipeline application:
```bash
streamlit run car_plate_ocr_app.py
```

This application provides:
- Full pipeline: Vehicle detection → Image enhancement → Plate detection → OCR
- Interactive interface with configurable parameters
- Support for Arabic and English license plates
- Multiple OCR engine comparison

#### Alternative Applications

**Vehicle Detection Only:**
```bash
streamlit run app.py
```

**License Plate Detection Only:**
```bash
streamlit run plate_detection_app.py
```

**Unified Detection (Both):**
```bash
streamlit run unified_detection_app.py
```

## 📁 Project Structure

```
CarDetection/
├── app.py                          # Vehicle detection only
├── car_plate_ocr_app.py           # Complete pipeline (MAIN APP)
├── plate_detection_app.py          # License plate detection only
├── unified_detection_app.py        # Unified detection interface
├── requirements.txt                # Python dependencies
├── OCR_INSTALLATION.md             # OCR setup guide
├── README.md                       # This file
│
├── car_detection_model.pt          # Vehicle detection model
├── yolo11n.pt                      # YOLO base model
│
├── train_car_detection_model.ipynb      # Training notebook for vehicles
├── train_plate_recognition_model.ipynb  # Training notebook for plates
│
├── runs/                           # Training outputs
│   └── detect/
│       ├── car_detection_yolo11/
│       └── plate_detection_yolo11/
│
└── plateRecignation/               # License plate dataset
    ├── train/
    └── test/
```

## 🎯 Usage Guide

### Using the Complete Pipeline

1. **Launch the application:**
   ```bash
   streamlit run car_plate_ocr_app.py
   ```

2. **Upload an image** containing one or more vehicles

3. **Step 1 - Car Detection:**
   - The system automatically detects all vehicles
   - View bounding boxes and confidence scores
   - Adjust confidence threshold if needed

4. **Step 2 - Image Enhancement:**
   - View original and enhanced vehicle images
   - Adjust enhancement parameters in the sidebar:
     - Brightness
     - Contrast
     - Sharpness
     - Denoising
     - CLAHE

5. **Step 3 - Select Vehicle:**
   - Choose which vehicle to process for license plate detection

6. **Step 4 - License Plate Detection:**
   - System detects license plates on the selected vehicle
   - View detected plates with bounding boxes

7. **Step 5 - OCR Recognition:**
   - System extracts text from detected plates
   - View results from all OCR engines
   - See the best result automatically selected

### Configuration Options

#### Image Enhancement
- **Brightness**: 0.5 - 2.0 (default: 1.2)
- **Contrast**: 0.5 - 2.0 (default: 1.3)
- **Sharpness**: 0.0 - 3.0 (default: 1.5)
- **Denoising**: Enable/disable
- **CLAHE**: Enable/disable contrast enhancement

#### Detection Thresholds
- **Car Confidence**: 0.0 - 1.0 (default: 0.25)
- **Plate Confidence**: 0.0 - 1.0 (default: 0.25)

#### OCR Settings
- **OCR Engine**: Choose between EasyOCR, PaddleOCR, or Tesseract
- **Preprocessing Method**: 
  - Auto (Try All)
  - Adaptive Threshold
  - OTSU Threshold
  - Morphological
  - Gaussian + OTSU
  - CLAHE + OTSU
- **Try All Engines**: Compare results from all available OCR engines

## 🛠️ Training Models

### Training Vehicle Detection Model

1. Open `train_car_detection_model.ipynb`
2. Ensure your dataset is in YOLO format
3. Configure training parameters
4. Run all cells to train the model
5. Model will be saved to `runs/detect/car_detection_yolo11/weights/best.pt`

### Training License Plate Detection Model

1. Open `train_plate_recognition_model.ipynb`
2. Ensure your dataset is in YOLO format
3. Configure training parameters
4. Run all cells to train the model
5. Model will be saved to `runs/detect/plate_detection_yolo11/weights/best.pt`

## 📊 Technical Details

### Models Used

- **YOLO (You Only Look Once)**: Object detection framework
- **YOLOv11**: Latest version for vehicle and plate detection

### Image Processing

- **OpenCV**: Image manipulation and preprocessing
- **PIL/Pillow**: Image enhancement
- **NumPy**: Numerical operations

### OCR Engines

1. **EasyOCR**
   - Languages: Arabic, English
   - GPU support available
   - Fast inference

2. **PaddleOCR**
   - Languages: Arabic, English
   - Recommended for Arabic/Tunisian plates
   - GPU support available
   - High accuracy

3. **Tesseract**
   - Languages: Arabic, English
   - Open-source
   - Requires system installation

## 🔧 Requirements

See `requirements.txt` for complete list. Main dependencies:

- `ultralytics>=8.0.0` - YOLO framework
- `streamlit>=1.28.0` - Web interface
- `opencv-python>=4.8.0` - Image processing
- `pillow>=10.0.0` - Image enhancement
- `numpy>=1.24.0` - Numerical operations
- `easyocr>=1.7.0` - OCR engine (optional)
- `paddleocr>=2.7.0` - OCR engine (optional)
- `pytesseract>=0.3.10` - Tesseract wrapper (optional)

## 🌍 Supported Languages

- **Arabic** (العربية)
- **English**
- **Tunisian License Plates** (combination of Arabic and English)

## 📝 License

This project is for educational purposes. Please ensure compliance with local regulations when using license plate recognition systems.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- YOLO team for the excellent object detection framework
- OCR engine developers (EasyOCR, PaddleOCR, Tesseract)
- Streamlit for the web framework

---

**Note**: This system is designed for educational and research purposes. Ensure you comply with privacy laws and regulations when processing license plate data.

