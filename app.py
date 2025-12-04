import streamlit as st
import cv2
import numpy as np
import os
import time
import tempfile

# ==========================================
# CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="Image Processing Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    /* Headers */
    h1, h2, h3 {
        color: #0e1117;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        font-weight: 700;
        border-bottom: 2px solid #4e8cff;
        padding-bottom: 10px;
    }
    /* Cards/Containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        color: #1976d2;
        border-bottom: 2px solid #1976d2;
    }
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #2c3e50;
    }
    /* Custom info box */
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2E86C1;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOGIC FUNCTIONS
# ==========================================

def is_prime(num):
    """Check if a number is prime."""
    if num <= 1: return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0: return False
    return True

def create_dummy_video(filename):
    """Generates a sample video for testing."""
    width, height = 640, 480
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(30): # 30 frames
        frame = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8) # Dark background
        
        # Grid lines for professional look
        cv2.line(frame, (0, 100), (640, 100), (50, 50, 50), 1)
        cv2.line(frame, (0, 300), (640, 300), (50, 50, 50), 1)
        
        # Moving Element
        cv2.rectangle(frame, (10 + i*15, 150), (110 + i*15, 250), (200, 200, 255), -1)
        cv2.putText(frame, "Sample Video", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Static Element
        cv2.circle(frame, (500, 350), 60, (0, 255, 127), -1)
        
        out.write(frame)
    out.release()
    return filename

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10005/10005937.png", width=60)
    st.title("Control Panel")
    
    st.subheader("1. Source Selection")
    option = st.radio("Input Source:", ["Generate Test Video", "Upload Video"], label_visibility="collapsed")
    
    st.divider()
    
    st.subheader("2. Settings")
    # Color customization
    highlight_color = st.color_picker("Highlight Color", "#00FF00")
    
    st.info("Use this sidebar to configure input and analysis parameters.")

# ==========================================
# MAIN APP
# ==========================================

st.title("Image Processing Workbench")
st.markdown('<div class="info-box">Welcome to the Practical 1 Analysis Tool. This application processes video input, extracts frames, and performs various computer vision transformations.</div>', unsafe_allow_html=True)

# --- Temp File Handling ---
tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
video_path = None

col1, col2 = st.columns([1, 2])

# --- Input Handling ---
with col1:
    st.markdown("### Input Video")
    if option == "Generate Test Video":
        if st.button("🚀 Generate & Load Video", type="primary", use_container_width=True):
            create_dummy_video(tfile.name)
            st.success("Video Generated!")
            video_path = tfile.name
        elif os.path.exists(tfile.name) and os.path.getsize(tfile.name) > 0:
             video_path = tfile.name
    else:
        uploaded_file = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

if video_path:
    with col1:
        st.video(video_path)
    
    # --- Frame Extraction ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        with col2:
            st.markdown("### Frame Analysis")
            frame_idx = st.slider("Select Frame to Analyze", 0, total_frames-1, 15)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Analysis Frame: {frame_idx}", width=400)
                img = frame # Store BGR for processing
            else:
                st.error("Error reading frame.")
                st.stop()
    cap.release()
    
    st.divider()

    # --- Analysis Tabs ---
    st.header("Processing Results")
    tabs = st.tabs([
        "📊 Properties", "👁️ Display", "⚫ B&W", 
        "🔄 Rotate", "🪞 Mirror", "🔍 Detection", 
        "✂️ Cuts", "🔢 Grid"
    ])

    # Q4: Properties (Moved first for dashboard feel)
    with tabs[0]:
        st.subheader("Image Metadata")
        h, w, c = img.shape
        file_stats = os.stat(video_path)
        size_mb = file_stats.st_size / (1024 * 1024)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Width", f"{w} px")
        m2.metric("Height", f"{h} px")
        m3.metric("Channels", f"{c}")
        m4.metric("File Size", f"{size_mb:.2f} MB")
        
        st.caption(f"File created: {time.ctime(file_stats.st_ctime)}")

    # Q2: Display
    with tabs[1]:
        st.subheader("Original Representation")
        st.image(frame_rgb, caption="Source Input")

    # Q3: B&W
    with tabs[2]:
        st.subheader("Grayscale Conversion")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c1, c2 = st.columns(2)
        c1.image(frame_rgb, caption="Original")
        c2.image(gray_img, caption="Grayscale (Black & White)")

    # Q5: Rotate
    with tabs[3]:
        st.subheader("Geometric Rotation")
        rot_col1, rot_col2 = st.columns([1, 3])
        with rot_col1:
            angle = st.radio("Rotation Angle", ["90° Clockwise", "180°", "270° (90° CCW)"])
        
        with rot_col2:
            if angle == "90° Clockwise":
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == "180°":
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            else:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            st.image(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    # Q6: Mirror
    with tabs[4]:
        st.subheader("Mirror Effect")
        mirror = cv2.flip(img, 1)
        st.image(cv2.cvtColor(mirror, cv2.COLOR_BGR2RGB), caption="Horizontal Flip")

    # Q7: Detection
    with tabs[5]:
        st.subheader("Object Detection")
        
        col_ctrl, col_view = st.columns([1, 3])
        with col_ctrl:
            thresh_val = st.slider("Threshold Sensitivity", 0, 255, 127)
            st.caption("Adjust to filter background noise.")
        
        with col_view:
            _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            detect_img = img.copy()
            count = 0
            
            # Convert user hex color to RGB tuple
            h_color = highlight_color.lstrip('#')
            rgb_color = tuple(int(h_color[i:i+2], 16) for i in (0, 2, 4))
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                    cv2.rectangle(detect_img, (x, y), (x + w_rect, y + h_rect), rgb_color, 2)
                    count += 1
            
            st.image(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB), caption=f"Found {count} Objects")

    # Q9/10: Cuts
    with tabs[6]:
        st.subheader("Image Segmentation")
        cut_type = st.selectbox("Select Cut Type", ["Vertical (80-20)", "Horizontal (70-30)"])
        
        if cut_type == "Vertical (80-20)":
            split_x = int(w * 0.8)
            col1, col2 = st.columns([4, 1])
            col1.image(cv2.cvtColor(img[:, :split_x], cv2.COLOR_BGR2RGB), caption="Left 80%")
            col2.image(cv2.cvtColor(img[:, split_x:], cv2.COLOR_BGR2RGB), caption="Right 20%")
        else:
            split_y = int(h * 0.7)
            st.image(cv2.cvtColor(img[:split_y, :], cv2.COLOR_BGR2RGB), caption="Top 70%")
            st.image(cv2.cvtColor(img[split_y:, :], cv2.COLOR_BGR2RGB), caption="Bottom 30%")

    # Q11: Grid
    with tabs[7]:
        st.subheader("Prime Number Grid")
        st.markdown("Generates a 10x10 grid and blacks out cells where the cell index is a **Prime Number**.")
        
        grid_img = img.copy()
        rows, cols = 10, 10
        step_h, step_w = h // rows, w // cols
        counter = 1
        
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * step_h, (r + 1) * step_h
                x1, x2 = c * step_w, (c + 1) * step_w
                
                if is_prime(counter):
                    grid_img[y1:y2, x1:x2] = 0
                
                # Grid Overlay
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(grid_img, str(counter), (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                counter += 1
        
        st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))

else:
    # Empty State
    st.info("👈 Please Select 'Generate Test Video' or Upload a file in the sidebar to begin.")
