import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    h1 {
        color: #333;
        text-align: center;
    }
    .uploaded-img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")
conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("### Model Properties")
st.sidebar.info("Model: YOLOv11n (Custom Trained)")
st.sidebar.info("Classes: Weapon, Knife, Pistol")

# --- App Header ---
st.title("🛡️ AI-Powered Weapon Detection")
st.markdown("### Upload an image or video to detect weapons instantly.")
st.markdown("---")

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    model = load_model("best (2).pt")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Main Content ---
uploaded_file = st.file_uploader("Choose a file...", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # Image Processing
    if file_type in ['png', 'jpg', 'jpeg']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📸 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("##### 🎯 Detection Results")
            if st.button("Analyze Image"):
                with st.spinner("Processing..."):
                    try:
                        # Perform inference
                        results = model.predict(image, conf=conf_thres, iou=iou_thres)
                        result = results[0]
                        
                        # Visualize results
                        res_plotted = result.plot()
                        res_image = Image.fromarray(res_plotted[..., ::-1]) # Convert BGR to RGB
                        st.image(res_image, use_container_width=True)
                        
                        # Stats
                        boxes = result.boxes
                        num_detections = len(boxes)
                        
                        if num_detections > 0:
                            st.success(f"⚠️ **Threat Detected!** Found **{num_detections}** potential weapon(s).")
                            # Detailed counts
                            class_counts = {}
                            for box in boxes:
                                cls = int(box.cls[0])
                                name = model.names[cls]
                                class_counts[name] = class_counts.get(name, 0) + 1
                            
                            st.markdown("#### Detailed Report:")
                            for name, count in class_counts.items():
                                st.write(f"- **{name.capitalize()}**: {count}")
                        else:
                            st.balloons()
                            st.success("✅ **Safe.** No weapons detected.")
                            
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        
    # Video Processing
    elif file_type in ['mp4', 'avi', 'mov']:
        st.markdown("##### 🎥 Video Analysis")
        
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
             st.video(tfile.name)
        
        with col2:
             if st.button("Start Video Verification"):
                st_frame = st.empty()
                cap = cv2.VideoCapture(tfile.name)
                
                stop_button = st.button("Stop Processing")
                
                while cap.isOpened():
                    if stop_button:
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    results = model.predict(frame, conf=conf_thres, iou=iou_thres)
                    res_plotted = results[0].plot()
                    
                    # Display
                    st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
                cap.release()
                st.success("Video processing complete.")
