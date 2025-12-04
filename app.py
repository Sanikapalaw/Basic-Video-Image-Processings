import streamlit as st
import cv2
import numpy as np
import os
import time
import tempfile

# ==========================================
# PAGE CONFIG & CSS STYLING
# ==========================================
st.set_page_config(page_title="Practical 1", page_icon="🎨", layout="wide")

# Custom CSS for a colorful look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    /* Headers */
    h1 {
        color: #FF4B4B;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2, h3 {
        color: #2E86C1;
    }
    /* Customizing Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f4f8;
        color: #2E86C1;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def is_prime(num):
    if num <= 1: return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0: return False
    return True

# Convert Hex color (e.g. #FF0000) to RGB Tuple (255, 0, 0)
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_dummy_video(filename):
    width, height = 640, 480
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(30): 
        # Create a colorful background (dark blue-ish)
        frame = np.full((height, width, 3), (50, 20, 20), dtype=np.uint8)
        
        # Moving White Square
        cv2.rectangle(frame, (10 + i*10, 100), (110 + i*10, 200), (255, 255, 255), -1)
        
        # Static Red Circle
        cv2.circle(frame, (400, 300), 50, (0, 0, 255), -1)
        
        # Static Yellow Triangle (Polygon)
        pts = np.array([[500, 100], [450, 200], [550, 200]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 255, 255))
        
        out.write(frame)
    out.release()
    return filename

# ==========================================
# SIDEBAR CONTROLS (COLOR SETTINGS)
# ==========================================
st.sidebar.title("🎨 Settings")
st.sidebar.write("Customize your output colors here!")

# Color Pickers
detect_color_hex = st.sidebar.color_picker("Object Detection Box Color", "#00FF00")
grid_line_color_hex = st.sidebar.color_picker("Grid Line Color", "#FF00FF")
text_color_hex = st.sidebar.color_picker("Text Color", "#FFFFFF")

# Convert Hex to RGB (Streamlit gives Hex, OpenCV needs RGB/BGR)
# Note: OpenCV uses BGR, but Streamlit images are RGB. 
# We will draw in RGB since we convert BGR->RGB before display.
detect_color = hex_to_rgb(detect_color_hex)
grid_line_color = hex_to_rgb(grid_line_color_hex)
text_color = hex_to_rgb(text_color_hex)

# ==========================================
# MAIN APP LOGIC
# ==========================================

st.title("Practical 1: Image Processing")
st.markdown("Processing Video ➝ Extracting Image ➝ Solving Q1-Q11")

# --- Step 1: Input Video ---
st.markdown("---")
st.header("1. Input Video Source")
col1, col2 = st.columns([1, 2])

with col1:
    option = st.radio("Choose Input Source:", ["Generate Dummy Video", "Upload Video"])

tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

with col2:
    if option == "Generate Dummy Video":
        if st.button("✨ Generate Test Video"):
            create_dummy_video(tfile.name)
            st.success("Video Generated Successfully!")
            st.video(tfile.name)
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile.write(uploaded_file.read())
            st.success("Video Uploaded")
            st.video(tfile.name)

# --- Step 2: Extract Image ---
if os.path.getsize(tfile.name) > 0:
    st.markdown("---")
    st.header("2. Extract Frame (Q1)")
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        col_slider, col_preview = st.columns([2, 1])
        
        with col_slider:
            frame_idx = st.slider("Select Frame Index to Cut", 0, total_frames-1, 15)
            st.info("Slide to choose a different moment from the video.")
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # OpenCV loads in BGR. We keep 'img' as BGR for processing logic,
            # but create 'img_rgb' for display.
            img = frame 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with col_preview:
                st.image(img_rgb, caption=f"Extracted Frame {frame_idx}", use_container_width=True)
        else:
            st.error("Could not read frame.")
            st.stop()
    else:
        st.warning("Please generate or upload a video first.")
        st.stop()
    cap.release()

    # --- Step 3: Practical Questions ---
    st.markdown("---")
    st.header("3. Practical Solutions")
    
    tabs = st.tabs([
        "Q2 Display", "Q3 B&W", "Q4 Props", 
        "Q5 Rotate", "Q6 Mirror", "Q7 Objects", 
        "Q9/10 Cuts", "Q11 Grid"
    ])

    # Q2: Show Image
    with tabs[0]:
        st.subheader("Q2: Original Image")
        st.image(img_rgb, use_container_width=True)

    # Q3: Black and White
    with tabs[1]:
        st.subheader("Q3: Grayscale Conversion")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(gray_img, caption="Grayscale", use_container_width=True)

    # Q4: Properties
    with tabs[2]:
        st.subheader("Q4: Image Properties")
        h, w, c = img.shape
        file_stats = os.stat(tfile.name)
        size_mb = file_stats.st_size / (1024 * 1024)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Width", f"{w} px")
        c2.metric("Height", f"{h} px")
        c3.metric("Channels", c)
        
        st.info(f"📁 Video File Size: **{size_mb:.2f} MB**")
        st.info(f"📅 Creation Date: **{time.ctime(file_stats.st_ctime)}**")

    # Q5: Rotate
    with tabs[3]:
        st.subheader("Q5: Rotation")
        rot_option = st.radio("Select Rotation:", ["90° Clockwise", "180°", "90° Counter-Clockwise"], horizontal=True)
        
        if rot_option == "90° Clockwise":
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot_option == "180°":
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        st.image(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Q6: Mirror
    with tabs[4]:
        st.subheader("Q6: Mirror Image")
        mirror = cv2.flip(img, 1)
        col_orig, col_mirror = st.columns(2)
        with col_orig:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with col_mirror:
            st.image(cv2.cvtColor(mirror, cv2.COLOR_BGR2RGB), caption="Mirrored", use_container_width=True)

    # Q7: Object Detection
    with tabs[5]:
        st.subheader("Q7: Object Detection")
        st.write("Detecting objects using thresholding and contours.")
        
        col_thresh, col_res = st.columns(2)
        
        thresh_val = col_thresh.slider("Threshold Sensitivity", 0, 255, 127)
        _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
        col_thresh.image(thresh, caption="Threshold Mask (Internal Step)", use_container_width=True)
        
        # Detection Logic
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Work on a copy of RGB image for drawing
        detect_img = img_rgb.copy()
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500: # Filter noise
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                # Draw using selected color from sidebar
                cv2.rectangle(detect_img, (x, y), (x + w_rect, y + h_rect), detect_color, 3)
                count += 1
        
        col_res.metric("Objects Found", count)
        col_res.image(detect_img, caption="Final Detection", use_container_width=True)

    # Q9 & Q10: Cuts
    with tabs[6]:
        st.subheader("Image Slicing")
        
        st.write("### Q9: Vertical Cut (80% Left | 20% Right)")
        split_x = int(w * 0.8)
        c1, c2 = st.columns([4, 1])
        c1.image(cv2.cvtColor(img[:, :split_x], cv2.COLOR_BGR2RGB), caption="Left 80%", use_container_width=True)
        c2.image(cv2.cvtColor(img[:, split_x:], cv2.COLOR_BGR2RGB), caption="Right 20%", use_container_width=True)

        st.markdown("---")
        
        st.write("### Q10: Horizontal Cut (70% Top | 30% Bottom)")
        split_y = int(h * 0.7)
        st.image(cv2.cvtColor(img[:split_y, :], cv2.COLOR_BGR2RGB), caption="Top 70%", use_container_width=True)
        st.image(cv2.cvtColor(img[split_y:, :], cv2.COLOR_BGR2RGB), caption="Bottom 30%", use_container_width=True)

    # Q11: Grid
    with tabs[7]:
        st.subheader("Q11: Prime Number Grid")
        st.write("Grid blocks where the counter is a Prime Number are removed (blacked out).")
        
        grid_img = img_rgb.copy()
        rows = 10
        cols = 10
        step_h = h // rows
        step_w = w // cols

        counter = 1
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * step_h, (r + 1) * step_h
                x1, x2 = c * step_w, (c + 1) * step_w
                
                # Logic: If prime, make it black
                if is_prime(counter):
                    grid_img[y1:y2, x1:x2] = 0 
                
                # Draw grid lines with sidebar color
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), grid_line_color, 1)
                
                # Draw text with sidebar color
                cv2.putText(grid_img, str(counter), (x1+5, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                
                counter += 1
        
        st.image(grid_img, caption="Grid Result", use_container_width=True)

else:
    st.info("👋 Waiting for video input...")
