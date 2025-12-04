import streamlit as st
import cv2
import numpy as np
import os
import time
import tempfile
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Unified Media Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (From App 2 for professional look)
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3 { color: #000000 !important; font-family: 'Helvetica Neue', sans-serif; }
    h1 { border-bottom: 2px solid #4e8cff; padding-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f8f9fa; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #e3f2fd; color: #000000 !important; border-bottom: 2px solid #1976d2; }
    [data-testid="stMetricValue"] { font-size: 1.2rem; color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def is_prime(num):
    """Check if a number is prime (From App 2)"""
    if num <= 1: return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0: return False
    return True

def load_image_from_upload(uploaded_file):
    """Decodes an image file to BGR (From App 1)"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

# ==========================================
# 3. SIDEBAR: INPUT SELECTION
# ==========================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # Mode Selector
    app_mode = st.radio("Select Source Type:", ["📷 Image Analysis", "🎥 Video Analysis"])
    st.divider()
    
    source_img = None  # This will hold the final BGR image to be processed
    video_path = None
    file_details = {}

    if app_mode == "📷 Image Analysis":
        st.info("Upload a static image (JPG/PNG).")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            source_img = load_image_from_upload(uploaded_file)
            file_details = {"name": uploaded_file.name, "size": uploaded_file.size}

    elif app_mode == "🎥 Video Analysis":
        st.info("Upload a video to extract frames.")
        uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            # Save temp file for OpenCV (From App 2)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            file_details = {"name": uploaded_file.name, "size": uploaded_file.size}

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
st.title("Unified Media Analytics Studio")

# If Video Mode: Handle Player and Frame Extraction FIRST
if app_mode == "🎥 Video Analysis" and video_path:
    col_vid1, col_vid2 = st.columns([1, 2])
    
    with col_vid1:
        st.video(video_path)
    
    with col_vid2:
        st.markdown("### Frame Selector")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            frame_idx = st.slider("Select Frame", 0, total_frames-1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                source_img = frame # Set the video frame as the source image
            cap.release()
        else:
            st.error("Could not read video frames.")

# ==========================================
# 5. PROCESSING LOGIC (Unified for Image & Video)
# ==========================================
if source_img is not None:
    # Convert BGR (OpenCV) to RGB (Streamlit/PIL) for consistent display
    img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    h, w, c = source_img.shape

    st.divider()
    
    # We use the Tab structure from App 2 as it handles many features better
    tabs = st.tabs([
        "📊 Overview", 
        "🛠️ Filters & Edges", 
        "🔄 Geometry", 
        "🕵️ AI Detection", 
        "✂️ Splitting",
        "🔢 Prime Grid"
    ])

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        st.subheader("Image Properties")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Width", f"{w} px")
        c2.metric("Height", f"{h} px")
        c3.metric("Channels", c)
        
        # Calculate size in MB
        size_mb = file_details.get("size", 0) / (1024 * 1024)
        c4.metric("File Size", f"{size_mb:.2f} MB")
        
        st.image(img_rgb, caption="Source Image", use_container_width=True)

    # --- TAB 2: FILTERS & EDGES (Combined App 1 & 2) ---
    with tabs[1]:
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.subheader("Grayscale")
            gray_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            st.image(gray_img, caption="Grayscale", use_container_width=True)

        with col_f2:
            st.subheader("Canny Edge Detection")
            st.caption("Adjust thresholds (From App 1)")
            t_lower = st.slider("Lower Threshold", 0, 255, 100)
            t_upper = st.slider("Upper Threshold", 0, 255, 200)
            edges = cv2.Canny(gray_img, t_lower, t_upper)
            st.image(edges, caption="Edge Map", use_container_width=True)

    # --- TAB 3: GEOMETRY ---
    with tabs[2]:
        st.subheader("Transformations")
        geo_col1, geo_col2 = st.columns([1, 2])
        
        with geo_col1:
            tr_type = st.radio("Transformation Type:", ["Rotate 90° Clockwise", "Rotate 180°", "Mirror (Flip)"])
        
        with geo_col2:
            if tr_type == "Rotate 90° Clockwise":
                processed = cv2.rotate(source_img, cv2.ROTATE_90_CLOCKWISE)
            elif tr_type == "Rotate 180°":
                processed = cv2.rotate(source_img, cv2.ROTATE_180)
            elif tr_type == "Mirror (Flip)":
                processed = cv2.flip(source_img, 1)
                
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- TAB 4: AI DETECTION (From App 2) ---
    with tabs[3]:
        st.subheader("Object Contours")
        
        d_col1, d_col2 = st.columns([1, 3])
        with d_col1:
            thresh_val = st.slider("Binarization Threshold", 0, 255, 127)
        
        with d_col2:
            gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            detect_img = source_img.copy()
            count = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 500: # Filter small noise
                    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                    cv2.rectangle(detect_img, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)
                    count += 1
            
            st.image(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB), caption=f"Detected {count} Objects")

    # --- TAB 5: SPLITTING (Combined Logic) ---
    with tabs[4]:
        st.subheader("Image Segmentation")
        split_mode = st.selectbox("Choose Split:", ["Vertical (80/20)", "Horizontal (70/30)"])
        
        if split_mode == "Vertical (80/20)":
            split = int(0.8 * w)
            left = img_rgb[:, :split]
            right = img_rgb[:, split:]
            sc1, sc2 = st.columns([4, 1])
            sc1.image(left, caption="Left 80%", use_container_width=True)
            sc2.image(right, caption="Right 20%", use_container_width=True)
            
        elif split_mode == "Horizontal (70/30)":
            split = int(0.7 * h)
            top = img_rgb[:split, :]
            bottom = img_rgb[split:, :]
            st.image(top, caption="Top 70%", use_container_width=True)
            st.image(bottom, caption="Bottom 30%", use_container_width=True)

    # --- TAB 6: PRIME GRID (Unique feature from App 2) ---
    with tabs[5]:
        st.subheader("Prime Number Grid Analysis")
        st.write("Overlays a 10x10 grid and blacks out cells where the index is a prime number.")
        
        grid_img = source_img.copy()
        rows, cols = 10, 10
        step_h, step_w = h // rows, w // cols
        counter = 1
        
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * step_h, (r + 1) * step_h
                x1, x2 = c * step_w, (c + 1) * step_w
                
                if is_prime(counter):
                    grid_img[y1:y2, x1:x2] = 0 # Black out prime cells
                
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(grid_img, str(counter), (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                counter += 1
        
        st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    # Empty State
    st.info("👈 Waiting for Upload. Please select a mode and upload a file in the sidebar.")
