import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO
from pathlib import Path
import io

# OCR imports (try to import, handle if not available)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Car & License Plate OCR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚗 Car Detection → License Plate OCR System")
st.markdown("""
**Complete Pipeline:** Upload an image → Detect cars → Enhance images → Select car → Detect license plate → Extract text (Arabic/Tunisian support)
""")

# Load models
@st.cache_resource
def load_car_model():
    """Load the YOLO car detection model"""
    model_path = Path("car_detection_model.pt")
    if model_path.exists():
        return YOLO(str(model_path))
    else:
        runs_model = Path("runs/detect/car_detection_yolo11/weights/best.pt")
        if runs_model.exists():
            return YOLO(str(runs_model))
        else:
            alt_model = Path("runs/detect/car_detection_yolo116/weights/best.pt")
            if alt_model.exists():
                return YOLO(str(alt_model))
    return None

@st.cache_resource
def load_plate_model():
    """Load the YOLO license plate detection model"""
    runs_model = Path("runs/detect/plate_detection_yolo11/weights/best.pt")
    if runs_model.exists():
        return YOLO(str(runs_model))
    else:
        alt_model = Path("runs/detect/plate_detection_yolo11/weights/last.pt")
        if alt_model.exists():
            return YOLO(str(alt_model))
    return None

@st.cache_resource
def load_easyocr_reader():
    """Load EasyOCR reader with Arabic and English support"""
    if not EASYOCR_AVAILABLE:
        return None
    try:
        reader = easyocr.Reader(['ar', 'en'], gpu=st.session_state.get('use_gpu', False))
        return reader
    except Exception as e:
        st.error(f"Error loading EasyOCR: {e}")
        return None

@st.cache_resource
def load_paddleocr_reader():
    """Load PaddleOCR reader with Arabic and English support"""
    if not PADDLEOCR_AVAILABLE:
        return None
    try:
        # PaddleOCR supports Arabic (ar) and English (en)
        ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=st.session_state.get('use_gpu', False))
        return ocr
    except Exception as e:
        st.error(f"Error loading PaddleOCR: {e}")
        return None

def run_easyocr(reader, image):
    """Run EasyOCR on image"""
    if reader is None:
        return []
    try:
        results = reader.readtext(image, paragraph=False)
        return results
    except Exception as e:
        st.warning(f"EasyOCR error: {e}")
        return []

def run_paddleocr(ocr, image):
    """Run PaddleOCR on image"""
    if ocr is None:
        return []
    try:
        # PaddleOCR returns results in different format
        results = ocr.ocr(image, cls=True)
        if results and results[0]:
            # Convert to similar format as EasyOCR: [(bbox, text, confidence)]
            formatted_results = []
            for line in results[0]:
                if line:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]
                    formatted_results.append((bbox, text, confidence))
            return formatted_results
        return []
    except Exception as e:
        st.warning(f"PaddleOCR error: {e}")
        return []

def run_tesseract(image):
    """Run Tesseract OCR on image"""
    if not TESSERACT_AVAILABLE:
        return []
    try:
        # Tesseract with Arabic and English
        custom_config = r'--oem 3 --psm 6 -l ara+eng'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Get detailed data with bounding boxes
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Confidence > 0
                text = data['text'][i].strip()
                if text:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    conf = float(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                    
                    # Format bbox similar to other OCRs
                    bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                    results.append((bbox, text, conf))
        
        return results
    except Exception as e:
        st.warning(f"Tesseract error: {e}")
        return []

# Initialize session state
if 'car_images' not in st.session_state:
    st.session_state.car_images = []
if 'enhanced_car_images' not in st.session_state:
    st.session_state.enhanced_car_images = []
if 'selected_car_idx' not in st.session_state:
    st.session_state.selected_car_idx = None

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Car detection settings
st.sidebar.subheader("Car Detection")
car_confidence = st.sidebar.slider(
    "Car Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# Image enhancement settings
st.sidebar.subheader("Image Enhancement")
enhance_brightness = st.sidebar.slider(
    "Brightness",
    min_value=0.5,
    max_value=2.0,
    value=1.2,
    step=0.1
)
enhance_contrast = st.sidebar.slider(
    "Contrast",
    min_value=0.5,
    max_value=2.0,
    value=1.3,
    step=0.1
)
enhance_sharpness = st.sidebar.slider(
    "Sharpness",
    min_value=0.0,
    max_value=3.0,
    value=1.5,
    step=0.1
)
apply_denoise = st.sidebar.checkbox("Apply Denoising", value=True)
apply_clahe = st.sidebar.checkbox("Apply CLAHE (Contrast Enhancement)", value=True)

# License plate detection settings
st.sidebar.subheader("License Plate Detection")
plate_confidence = st.sidebar.slider(
    "Plate Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# OCR settings
st.sidebar.subheader("OCR Settings")

# Determine available OCR engines
available_engines = []
if EASYOCR_AVAILABLE:
    available_engines.append("EasyOCR")
if PADDLEOCR_AVAILABLE:
    available_engines.append("PaddleOCR")
if TESSERACT_AVAILABLE:
    available_engines.append("Tesseract")

# Initialize OCR variables
ocr_engine = None
try_all_engines = False
use_gpu = False

if not available_engines:
    st.sidebar.error("⚠️ No OCR engines available! Please install at least one:")
    st.sidebar.code("pip install easyocr\n# OR\npip install paddleocr\n# OR\npip install pytesseract")
else:
    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        available_engines,
        index=0 if "PaddleOCR" in available_engines else 0,  # Prefer PaddleOCR if available
        help="PaddleOCR is recommended for Arabic/Tunisian plates"
    )
    
    use_gpu = st.sidebar.checkbox("Use GPU for OCR (if available)", value=False)
    st.session_state.use_gpu = use_gpu
    
    # Option to try all engines and compare
    try_all_engines = st.sidebar.checkbox("Try All Available Engines (Compare Results)", value=False)

ocr_preprocessing_method = st.sidebar.selectbox(
    "OCR Preprocessing Method",
    ["Auto (Try All)", "Adaptive Threshold", "OTSU Threshold", "Morphological", "Gaussian + OTSU", "CLAHE + OTSU"],
    index=0,
    help="Choose preprocessing method for better OCR results"
)

min_plate_width = st.sidebar.slider(
    "Min Plate Width (pixels)",
    min_value=50,
    max_value=500,
    value=150,
    step=10,
    help="Minimum width for plate before resizing"
)

min_plate_height = st.sidebar.slider(
    "Min Plate Height (pixels)",
    min_value=30,
    max_value=200,
    value=60,
    step=5,
    help="Minimum height for plate before resizing"
)

# Load models
car_model = load_car_model()
if car_model is None:
    st.error("⚠️ Car detection model not found! Please train the model first.")
    st.stop()

plate_model = load_plate_model()
if plate_model is None:
    st.error("⚠️ License plate detection model not found! Please train the model first.")
    st.stop()

# Image enhancement functions
def enhance_image(image, brightness=1.2, contrast=1.3, sharpness=1.5, denoise=True, clahe=True):
    """Apply various image enhancements"""
    # Convert to PIL for enhancement
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = image
    
    # Apply brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    
    # Apply contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    
    # Apply sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness)
    
    # Convert back to OpenCV format
    img_array = np.array(pil_image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Apply denoising
    if denoise:
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if clahe:
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe_obj.apply(l)
        img_cv = cv2.merge([l, a, b])
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2BGR)
    
    return img_cv

def preprocess_plate_for_ocr(plate_img, method="auto", min_width=150, min_height=60):
    """Advanced preprocessing for license plate OCR"""
    preprocessed_images = {}
    
    # Normalize method name - handle "Auto (Try All)" as auto
    is_auto = method.lower() == "auto" or method == "Auto (Try All)"
    
    # Convert to grayscale
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    # Resize if too small
    h, w = gray.shape
    if w < min_width or h < min_height:
        scale_w = min_width / w if w < min_width else 1.0
        scale_h = min_height / h if h < min_height else 1.0
        scale = max(scale_w, scale_h) * 1.5  # Extra scaling for better OCR
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Method 1: OTSU Threshold
    if is_auto or method == "OTSU Threshold":
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images["OTSU Threshold"] = thresh_otsu
    
    # Method 2: Adaptive Threshold
    if is_auto or method == "Adaptive Threshold":
        thresh_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images["Adaptive Threshold"] = thresh_adaptive
    
    # Method 3: Morphological Operations
    if is_auto or method == "Morphological":
        # Apply OTSU first
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        preprocessed_images["Morphological"] = morph
    
    # Method 4: Gaussian Blur + OTSU
    if is_auto or method == "Gaussian + OTSU":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh_gauss = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images["Gaussian + OTSU"] = thresh_gauss
    
    # Method 5: CLAHE + OTSU
    if is_auto or method == "CLAHE + OTSU":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        _, thresh_clahe = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images["CLAHE + OTSU"] = thresh_clahe
    
    # If auto, return all; otherwise return the selected one
    if is_auto:
        return preprocessed_images
    else:
        if method in preprocessed_images:
            return {method: preprocessed_images[method]}
        else:
            # Fallback to OTSU if method not found
            return {"OTSU Threshold": preprocessed_images.get("OTSU Threshold", gray)}

def improve_plate_contrast(plate_img):
    """Additional contrast improvement for license plates"""
    # Convert to LAB color space
    lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Additional sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

# File uploader
uploaded_file = st.file_uploader(
    "📸 Upload an image with cars...",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload an image containing cars"
)

if uploaded_file is not None:
    # Read and display original image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert PIL to OpenCV format (RGB to BGR)
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array
    
    # STEP 1: Detect cars
    st.header("Step 1: 🚗 Car Detection")
    with st.spinner("Detecting cars in the image..."):
        results = car_model(img_cv, conf=car_confidence)
    
    # Process car detections
    car_detections = []
    annotated_img = img_cv.copy()
    target_class = "car"
    class_id = 1
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = car_model.names[cls]
            
            # Only process car detections
            if cls == class_id and class_name == target_class:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                car_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': class_name
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"Car {len(car_detections)}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 15), 
                             (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated_img_rgb, caption=f"Detected {len(car_detections)} car(s)", use_container_width=True)
    with col2:
        st.metric("Cars Detected", len(car_detections))
        if car_detections:
            avg_conf = np.mean([d['confidence'] for d in car_detections])
            st.metric("Average Confidence", f"{avg_conf:.2%}")
    
    if not car_detections:
        st.warning("No cars detected. Try adjusting the confidence threshold or upload a different image.")
        st.stop()
    
    # STEP 2: Extract and enhance car images
    st.header("Step 2: 🔍 Extract & Enhance Car Images")
    
    with st.spinner("Extracting and enhancing car images..."):
        car_images = []
        enhanced_car_images = []
        
        for i, det in enumerate(car_detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Extract car with padding
            padding = 20
            h, w = img_cv.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            # Crop the car
            car_crop = img_cv[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            car_images.append(car_crop)
            
            # Enhance the image
            enhanced = enhance_image(
                car_crop,
                brightness=enhance_brightness,
                contrast=enhance_contrast,
                sharpness=enhance_sharpness,
                denoise=apply_denoise,
                clahe=apply_clahe
            )
            enhanced_car_images.append(enhanced)
        
        st.session_state.car_images = car_images
        st.session_state.enhanced_car_images = enhanced_car_images
    
    # Display original and enhanced car images
    st.subheader("Original vs Enhanced Car Images")
    
    for i in range(len(car_images)):
        cols = st.columns(2)
        with cols[0]:
            car_rgb = cv2.cvtColor(car_images[i], cv2.COLOR_BGR2RGB)
            st.image(car_rgb, caption=f"Car #{i+1} - Original", use_container_width=True)
        with cols[1]:
            enhanced_rgb = cv2.cvtColor(enhanced_car_images[i], cv2.COLOR_BGR2RGB)
            st.image(enhanced_rgb, caption=f"Car #{i+1} - Enhanced", use_container_width=True)
    
    # STEP 3: Select car for license plate detection
    st.header("Step 3: 🎯 Select Car for License Plate Detection")
    
    car_options = [f"Car #{i+1} (Confidence: {car_detections[i]['confidence']:.2%})" 
                   for i in range(len(car_detections))]
    
    selected_car = st.selectbox(
        "Choose which car to detect the license plate on:",
        options=car_options,
        index=st.session_state.selected_car_idx if st.session_state.selected_car_idx is not None else 0
    )
    
    selected_idx = car_options.index(selected_car)
    st.session_state.selected_car_idx = selected_idx
    
    # Display selected car
    selected_car_img = enhanced_car_images[selected_idx]
    selected_car_rgb = cv2.cvtColor(selected_car_img, cv2.COLOR_BGR2RGB)
    st.image(selected_car_rgb, caption=f"Selected: {selected_car}", use_container_width=True)
    
    # STEP 4: Detect license plate
    st.header("Step 4: 🚙 License Plate Detection")
    
    with st.spinner("Detecting license plate on selected car..."):
        plate_results = plate_model(selected_car_img, conf=plate_confidence)
    
    plate_detections = []
    plate_annotated = selected_car_img.copy()
    
    for r in plate_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            plate_detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf
            })
            
            # Draw bounding box
            cv2.rectangle(plate_annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            label = f"Plate: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(plate_annotated, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), (255, 0, 0), -1)
            cv2.putText(plate_annotated, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    plate_annotated_rgb = cv2.cvtColor(plate_annotated, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(plate_annotated_rgb, caption="License Plate Detection", use_container_width=True)
    with col2:
        st.metric("Plates Detected", len(plate_detections))
        if plate_detections:
            avg_conf = np.mean([d['confidence'] for d in plate_detections])
            st.metric("Average Confidence", f"{avg_conf:.2%}")
    
    if not plate_detections:
        st.warning("No license plates detected. Try adjusting the confidence threshold or select a different car.")
    else:
        # STEP 5: OCR - Extract text from license plate
        st.header("Step 5: 📝 License Plate OCR (Arabic/English)")
        
        # Determine which OCR engines to use
        engines_to_try = []
        if try_all_engines:
            if EASYOCR_AVAILABLE:
                engines_to_try.append("EasyOCR")
            if PADDLEOCR_AVAILABLE:
                engines_to_try.append("PaddleOCR")
            if TESSERACT_AVAILABLE:
                engines_to_try.append("Tesseract")
        else:
            engines_to_try = [ocr_engine]
        
        if not engines_to_try:
            st.error("No OCR engines available! Please install at least one OCR library.")
            st.stop()
        
        # Load OCR readers
        easyocr_reader = None
        paddleocr_reader = None
        
        if "EasyOCR" in engines_to_try:
            with st.spinner("Loading EasyOCR model (this may take a moment on first run)..."):
                easyocr_reader = load_easyocr_reader()
        
        if "PaddleOCR" in engines_to_try:
            with st.spinner("Loading PaddleOCR model (this may take a moment on first run)..."):
                paddleocr_reader = load_paddleocr_reader()
        
        if not any([easyocr_reader, paddleocr_reader]) and "Tesseract" not in engines_to_try:
            st.error("Failed to load OCR models. Please check OCR installation.")
        else:
            # Process each detected plate
            all_ocr_results = []
            
            for i, plate_det in enumerate(plate_detections):
                x1, y1, x2, y2 = plate_det['bbox']
                
                # Extract plate region with padding
                padding = 10
                h, w = selected_car_img.shape[:2]
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x2 + padding)
                y2_pad = min(h, y2 + padding)
                
                # Crop the license plate
                plate_crop = selected_car_img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                
                # Further enhance the plate image for OCR
                plate_enhanced = improve_plate_contrast(plate_crop)
                plate_enhanced = enhance_image(
                    plate_enhanced,
                    brightness=1.5,
                    contrast=2.0,
                    sharpness=2.5,
                    denoise=True,
                    clahe=True
                )
                
                # Get preprocessed versions
                preprocessed_versions = preprocess_plate_for_ocr(
                    plate_enhanced, 
                    method=ocr_preprocessing_method,
                    min_width=min_plate_width,
                    min_height=min_plate_height
                )
                
                # Display plate images
                num_versions = len(preprocessed_versions)
                cols = st.columns(min(4, num_versions + 2))
                
                with cols[0]:
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    st.image(plate_rgb, caption=f"Plate #{i+1} - Original", use_container_width=True)
                
                with cols[1]:
                    plate_enhanced_rgb = cv2.cvtColor(plate_enhanced, cv2.COLOR_BGR2RGB)
                    st.image(plate_enhanced_rgb, caption=f"Plate #{i+1} - Enhanced", use_container_width=True)
                
                # Display preprocessed versions
                for idx, (method_name, processed_img) in enumerate(preprocessed_versions.items()):
                    if idx + 2 < len(cols):
                        with cols[idx + 2]:
                            st.image(processed_img, caption=f"{method_name}", use_container_width=True)
                
                # Run OCR on all preprocessed versions and all engines
                all_engine_results = {}
                best_overall_result = None
                best_overall_confidence = 0.0
                
                # Try each OCR engine
                for engine_name in engines_to_try:
                    engine_results = []
                    
                    for method_name, processed_img in preprocessed_versions.items():
                        with st.spinner(f"Running {engine_name} on plate #{i+1} ({method_name})..."):
                            try:
                                ocr_results = []
                                
                                # Run appropriate OCR engine
                                if engine_name == "EasyOCR" and easyocr_reader:
                                    ocr_results = run_easyocr(easyocr_reader, processed_img)
                                elif engine_name == "PaddleOCR" and paddleocr_reader:
                                    ocr_results = run_paddleocr(paddleocr_reader, processed_img)
                                elif engine_name == "Tesseract":
                                    ocr_results = run_tesseract(processed_img)
                                
                                if ocr_results:
                                    # Extract text and confidence
                                    texts = []
                                    confidences = []
                                    for (bbox, text, conf) in ocr_results:
                                        texts.append(text.strip())
                                        confidences.append(conf)
                                    
                                    combined_text = " ".join(texts)
                                    avg_ocr_conf = np.mean(confidences) if confidences else 0.0
                                    
                                    method_result = {
                                        'preprocessing': method_name,
                                        'engine': engine_name,
                                        'text': combined_text,
                                        'confidence': avg_ocr_conf,
                                        'individual_results': ocr_results
                                    }
                                    engine_results.append(method_result)
                                    
                                    # Track best overall result
                                    if avg_ocr_conf > best_overall_confidence:
                                        best_overall_confidence = avg_ocr_conf
                                        best_overall_result = method_result
                            except Exception as e:
                                st.warning(f"{engine_name} error for {method_name}: {str(e)}")
                    
                    all_engine_results[engine_name] = engine_results
                
                # Store results
                if best_overall_result:
                    all_ocr_results.append({
                        'plate_num': i + 1,
                        'text': best_overall_result['text'],
                        'confidence': best_overall_result['confidence'],
                        'engine': best_overall_result['engine'],
                        'preprocessing': best_overall_result['preprocessing'],
                        'all_engines': all_engine_results,
                        'individual_results': best_overall_result['individual_results']
                    })
                else:
                    all_ocr_results.append({
                        'plate_num': i + 1,
                        'text': "No text detected",
                        'confidence': 0.0,
                        'engine': "None",
                        'preprocessing': "None",
                        'all_engines': all_engine_results,
                        'individual_results': []
                    })
            
            # Display OCR results
            st.subheader("📋 OCR Results")
            
            for result in all_ocr_results:
                title = f"Plate #{result['plate_num']} - Best: {result['engine']} + {result['preprocessing']} (Confidence: {result['confidence']:.2%})"
                with st.expander(title, expanded=True):
                    st.markdown(f"### **Detected Text:** `{result['text']}`")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best OCR Confidence", f"{result['confidence']:.2%}")
                    with col2:
                        st.metric("Best Engine", result['engine'])
                    st.metric("Best Preprocessing", result['preprocessing'])
                    
                    if result['individual_results']:
                        st.write("**Individual character/word detections:**")
                        for idx, (bbox, text, conf) in enumerate(result['individual_results']):
                            st.write(f"- `{text}` (Confidence: {conf:.2%})")
                    
                    # Show results from all engines if multiple were tried
                    if len(result.get('all_engines', {})) > 1 or try_all_engines:
                        st.write("**Results from all OCR engines:**")
                        for engine_name, engine_results in result['all_engines'].items():
                            if engine_results:
                                st.write(f"**{engine_name}:**")
                                for method_result in engine_results:
                                    st.write(f"  - {method_result['preprocessing']}: `{method_result['text']}` (Confidence: {method_result['confidence']:.2%})")
                            else:
                                st.write(f"**{engine_name}:** No results")
                    
                    # Show all preprocessing method results for the best engine
                    best_engine_results = result.get('all_engines', {}).get(result['engine'], [])
                    if len(best_engine_results) > 1:
                        st.write(f"**All preprocessing methods for {result['engine']}:**")
                        for method_result in best_engine_results:
                            st.write(f"- {method_result['preprocessing']}: `{method_result['text']}` (Confidence: {method_result['confidence']:.2%})")
            
            # Summary
            if all_ocr_results:
                st.success("✅ OCR Processing Complete!")
                detected_texts = [r['text'] for r in all_ocr_results if r['text'] != "No text detected"]
                if detected_texts:
                    st.markdown("### **Summary:**")
                    for i, text in enumerate(detected_texts, 1):
                        st.markdown(f"**License Plate {i}:** `{text}`")

else:
    st.info("👆 Please upload an image to get started!")
    st.markdown("""
    ### 📖 How to use:
    1. **Upload Image**: Upload an image containing one or more cars
    2. **Car Detection**: The app will automatically detect all cars in the image
    3. **Image Enhancement**: View original and enhanced versions of each detected car
    4. **Select Car**: Choose which car you want to process for license plate detection
    5. **License Plate Detection**: The app will detect license plates on the selected car
    6. **OCR**: Extract text from detected license plates (supports Arabic and English)
    
    ### ⚙️ Tips:
    - Adjust confidence thresholds in the sidebar if detections are not accurate
    - Tune image enhancement parameters for better OCR results
    - The OCR model supports Arabic and English characters (Tunisian license plates)
    - First OCR run may take longer as the model downloads
    """)

