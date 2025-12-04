import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import io

# Page configuration
st.set_page_config(
    page_title="Vehicle Detection Suite",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🚗 Vehicle Detection Suite")
page = st.sidebar.radio(
    "Choose Detection Mode",
    ["Car Detection", "License Plate Detection"],
    index=0
)

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

# CAR DETECTION PAGE
if page == "Car Detection":
    st.title("🚗 Car Detection")
    st.markdown("Upload an image to detect cars and view zoomed-in views of each detected car.")
    
    car_model = load_car_model()
    if car_model is None:
        st.error("⚠️ Car detection model not found! Please train the model first using the notebook.")
        st.info("Expected paths:")
        st.info("- `car_detection_model.pt`")
        st.info("- `runs/detect/car_detection_yolo11/weights/best.pt`")
        st.info("- `runs/detect/car_detection_yolo116/weights/best.pt`")
        st.stop()
    
    # Sidebar settings for car detection
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Class filter - only show 'car' class (class index 1)
    target_class = "car"
    class_id = 1  # Car class index from the dataset
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
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
        
        # Run inference
        with st.spinner("Detecting cars..."):
            results = car_model(img_cv, conf=confidence_threshold)
        
        # Process results
        detections = []
        annotated_img = img_cv.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = car_model.names[cls]
                
                # Only process car detections
                if cls == class_id and class_name == target_class:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': class_name
                    })
                    
                    # Draw bounding box on image
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_img, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Convert back to RGB for display
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Display results
        st.subheader("Detection Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(annotated_img_rgb, caption=f"Detected {len(detections)} car(s)", use_container_width=True)
        
        with col2:
            st.metric("Cars Detected", len(detections))
            if detections:
                avg_confidence = np.mean([d['confidence'] for d in detections])
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
        
        # Display zoomed-in views of detected cars
        if detections:
            st.subheader("🔍 Zoomed-in Views of Detected Cars")
            
            # Create columns for car images (2 per row)
            num_cols = 2
            num_rows = (len(detections) + num_cols - 1) // num_cols
            
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    det_idx = row * num_cols + col_idx
                    if det_idx < len(detections):
                        det = detections[det_idx]
                        x1, y1, x2, y2 = det['bbox']
                        
                        # Extract car region with some padding
                        padding = 10
                        h, w = img_cv.shape[:2]
                        x1_pad = max(0, x1 - padding)
                        y1_pad = max(0, y1 - padding)
                        x2_pad = min(w, x2 + padding)
                        y2_pad = min(h, y2 + padding)
                        
                        # Crop the car
                        car_crop = img_cv[y1_pad:y2_pad, x1_pad:x2_pad]
                        car_crop_rgb = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
                        
                        # Display in column
                        with cols[col_idx]:
                            st.image(
                                car_crop_rgb,
                                caption=f"Car #{det_idx + 1} (Confidence: {det['confidence']:.2%})",
                                use_container_width=True
                            )
                            st.caption(f"BBox: ({x1}, {y1}) to ({x2}, {y2})")
        else:
            st.info("No cars detected in this image. Try adjusting the confidence threshold or upload a different image.")
        
        # Display detection details
        if detections:
            with st.expander("📊 Detection Details"):
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    st.write(f"**Car #{i + 1}**")
                    st.write(f"- Confidence: {det['confidence']:.2%}")
                    st.write(f"- Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
                    st.write(f"- Size: {x2 - x1} × {y2 - y1} pixels")
                    st.write("---")
    
    else:
        st.info("👆 Please upload an image to get started!")

# LICENSE PLATE DETECTION PAGE
elif page == "License Plate Detection":
    st.title("🚗 License Plate Detection")
    st.markdown("Upload an image to detect license plates using YOLOv11 model.")
    
    plate_model = load_plate_model()
    if plate_model is None:
        st.error("⚠️ License plate detection model not found! Please train the model first using the notebook.")
        st.info("Expected path: `runs/detect/plate_detection_yolo11/weights/best.pt`")
        st.stop()
    
    # Sidebar settings for plate detection
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression"
    )
    
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
    show_bbox_coords = st.sidebar.checkbox("Show Bounding Box Coordinates", value=False)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing license plates"
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
        
        # Run inference
        with st.spinner("🔍 Detecting license plates..."):
            results = plate_model(img_cv, conf=confidence_threshold, iou=iou_threshold)
        
        # Process results
        detections = []
        annotated_img = img_cv.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = plate_model.names[cls] if cls < len(plate_model.names) else "license_plate"
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': class_name
                })
                
                # Draw bounding box on image
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
                
                if show_confidence:
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # Background rectangle for text
                    cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 15), 
                                 (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(annotated_img, label, (x1 + 5, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Convert back to RGB for display
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Display results
        st.subheader("📊 Detection Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(annotated_img_rgb, caption=f"Detected {len(detections)} license plate(s)", use_container_width=True)
        
        with col2:
            st.metric("License Plates Detected", len(detections))
            if detections:
                avg_confidence = np.mean([d['confidence'] for d in detections])
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
                max_confidence = max([d['confidence'] for d in detections])
                st.metric("Max Confidence", f"{max_confidence:.2%}")
            else:
                st.info("No license plates detected. Try adjusting the confidence threshold.")
        
        # Display zoomed-in views of detected license plates
        if detections:
            st.subheader("🔍 Zoomed-in Views of Detected License Plates")
            
            # Create columns for plate images (2 per row)
            num_cols = 2
            num_rows = (len(detections) + num_cols - 1) // num_cols
            
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    det_idx = row * num_cols + col_idx
                    if det_idx < len(detections):
                        det = detections[det_idx]
                        x1, y1, x2, y2 = det['bbox']
                        
                        # Extract plate region with some padding
                        padding = 15
                        h, w = img_cv.shape[:2]
                        x1_pad = max(0, x1 - padding)
                        y1_pad = max(0, y1 - padding)
                        x2_pad = min(w, x2 + padding)
                        y2_pad = min(h, y2 + padding)
                        
                        # Crop the license plate
                        plate_crop = img_cv[y1_pad:y2_pad, x1_pad:x2_pad]
                        plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                        
                        # Display in column
                        with cols[col_idx]:
                            st.image(
                                plate_crop_rgb,
                                caption=f"Plate #{det_idx + 1} (Confidence: {det['confidence']:.2%})",
                                use_container_width=True
                            )
                            if show_bbox_coords:
                                st.caption(f"BBox: ({x1}, {y1}) to ({x2}, {y2})")
                                st.caption(f"Size: {x2 - x1} × {y2 - y1} pixels")
        else:
            st.info("No license plates detected in this image. Try adjusting the confidence threshold or upload a different image.")
        
        # Display detection details
        if detections:
            with st.expander("📋 Detection Details"):
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    st.write(f"**License Plate #{i + 1}**")
                    st.write(f"- Confidence: {det['confidence']:.2%}")
                    st.write(f"- Class: {det['class']}")
                    st.write(f"- Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
                    st.write(f"- Size: {x2 - x1} × {y2 - y1} pixels")
                    st.write(f"- Area: {(x2 - x1) * (y2 - y1):,} pixels²")
                    st.write("---")
        
        # Download annotated image
        if detections:
            st.subheader("💾 Download Results")
            # Convert annotated image to bytes
            annotated_pil = Image.fromarray(annotated_img_rgb)
            buf = io.BytesIO()
            annotated_pil.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Annotated Image",
                data=buf,
                file_name="license_plate_detection.png",
                mime="image/png"
            )
    
    else:
        st.info("👆 Please upload an image to get started!")
        st.markdown("""
        ### How to use:
        1. Upload an image using the file uploader above
        2. Adjust the confidence threshold if needed (default: 0.25)
        3. View the detection results and zoomed-in license plate views
        4. Download the annotated image if desired
        
        ### Tips:
        - Lower confidence threshold for more detections (may include false positives)
        - Higher confidence threshold for more accurate detections (may miss some plates)
        - The model works best with clear, well-lit images
        """)

