import streamlit as st
import cv2
import numpy as np
import tempfile
from moviepy.editor import VideoFileClip

# ==========================================
# 1. THE REUSABLE IMAGE ANALYSIS ENGINE
# ==========================================
def analyze_image(img_bgr):
    """
    This function contains all the 'Image Processing Assignment' logic (Q1-Q11).
    We can call this function for an uploaded photo OR a video frame.
    """
    st.markdown("---")
    st.subheader("🎨 Image Analysis Studio")
    
    # Menu for Image Tasks
    task = st.selectbox("Choose Analysis Task:", [
        "View Properties", 
        "Grayscale & Threshold", 
        "Geometric (Rotate/Flip)", 
        "Edge Detection (Canny)",
        "Splitting & Grid"
    ], key="img_task_menu")

    # --- TASK 1: PROPERTIES ---
    if task == "View Properties":
        h, w, c = img_bgr.shape
        st.write(f"**Dimensions:** {w} x {h}")
        st.write(f"**Channels:** {c}")
        st.write(f"**Total Pixels:** {w * h:,}")
        st.image(img_bgr, channels="BGR", caption="Current Frame/Image")

    # --- TASK 2: GRAYSCALE ---
    elif task == "Grayscale & Threshold":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        col1, col2 = st.columns(2)
        col1.image(gray, caption="Grayscale", use_container_width=True)
        
        # Binary Threshold
        thresh_val = st.slider("Threshold Value", 0, 255, 127)
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        col2.image(binary, caption="Binary (Black & White)", use_container_width=True)

    # --- TASK 3: GEOMETRIC ---
    elif task == "Geometric (Rotate/Flip)":
        geo_option = st.radio("Action:", ["Rotate 90° Clockwise", "Rotate 180°", "Mirror (Flip)"])
        if geo_option == "Rotate 90° Clockwise":
            processed = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif geo_option == "Rotate 180°":
            processed = cv2.rotate(img_bgr, cv2.ROTATE_180)
        else:
            processed = cv2.flip(img_bgr, 1)
        st.image(processed, channels="BGR")

    # --- TASK 4: EDGES ---
    elif task == "Edge Detection (Canny)":
        t1 = st.slider("Min Threshold", 0, 255, 100)
        t2 = st.slider("Max Threshold", 0, 255, 200)
        edges = cv2.Canny(img_bgr, t1, t2)
        st.image(edges, caption="Edges Detected")

    # --- TASK 5: SPLIT ---
    elif task == "Splitting & Grid":
        h, w = img_bgr.shape[:2]
        split_type = st.radio("Split Type:", ["Vertical 50/50", "Grid 2x2"])
        
        if split_type == "Vertical 50/50":
            mid = w // 2
            left = img_bgr[:, :mid]
            right = img_bgr[:, mid:]
            c1, c2 = st.columns(2)
            c1.image(left, channels="BGR", caption="Left Half")
            c2.image(right, channels="BGR", caption="Right Half")
        else:
            mid_h, mid_w = h // 2, w // 2
            top_left = img_bgr[:mid_h, :mid_w]
            top_right = img_bgr[:mid_h, mid_w:]
            bot_left = img_bgr[mid_h:, :mid_w]
            bot_right = img_bgr[mid_h:, mid_w:]
            
            c1, c2 = st.columns(2)
            c1.image(top_left, channels="BGR", caption="Top-Left")
            c1.image(bot_left, channels="BGR", caption="Bottom-Left")
            c2.image(top_right, channels="BGR", caption="Top-Right")
            c2.image(bot_right, channels="BGR", caption="Bottom-Right")


# ==========================================
# 2. MAIN APP
# ==========================================
st.set_page_config(page_title="Media Lab Pro", layout="wide")
st.title("🎬 Multimedia Processing Lab")

# Sidebar for Navigation
app_mode = st.sidebar.selectbox("Select Input Type:", ["Upload Video (Advanced)", "Upload Image (Simple)"])

# -------------------------------------------
# MODE A: VIDEO PROCESSING (The New Hybrid)
# -------------------------------------------
if app_mode == "Upload Video (Advanced)":
    st.markdown("### 🎥 Video Workflow")
    st.info("Upload a video -> Extract Audio -> **Extract a Frame to Analyze**")
    
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    
    if video_file:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        path = tfile.name
        
        # Get Video Info
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        # Create Tabs for neat organization
        tab1, tab2, tab3 = st.tabs(["1. Audio & Playback", "2. Frame Extractor", "3. Deep Analysis"])

        # --- TAB 1: AUDIO ---
        with tab1:
            st.write(f"**Duration:** {duration:.2f}s | **FPS:** {fps:.2f}")
            if st.button("Extract Audio"):
                clip = VideoFileClip(path)
                if clip.audio:
                    clip.audio.write_audiofile("audio.mp3", logger=None)
                    st.audio("audio.mp3")
                else:
                    st.warning("No audio found.")
            
            # Simple Video Player
            st.video(video_file)

        # --- TAB 2 & 3: FRAME EXTRACTION & ANALYSIS ---
        with tab2:
            st.header("📸 Extract a Frame")
            st.write("Slide to pick a specific second from the video.")
            
            # Slider to pick timestamp
            time_select = st.slider("Select Time (seconds)", 0.0, duration, 0.0, step=0.1)
            
            # Button to Grab Frame
            if st.button("Capture This Frame"):
                cap = cv2.VideoCapture(path)
                # Jump to specific time (msec)
                cap.set(cv2.CAP_PROP_POS_MSEC, time_select * 1000)
                success, frame = cap.read()
                cap.release()
                
                if success:
                    st.success(f"Captured frame at {time_select} seconds!")
                    st.image(frame, channels="BGR", caption="Captured Frame")
                    
                    # SAVE FRAME TO SESSION STATE
                    # (This lets us pass it to Tab 3 without losing it)
                    st.session_state['captured_frame'] = frame
                else:
                    st.error("Could not capture frame.")

        with tab3:
            # Check if we have a frame in memory
            if 'captured_frame' in st.session_state:
                st.header("🔬 Deep Analysis")
                st.markdown("Applying Image Processing Logic to the **Video Frame**.")
                
                # CALL THE ANALYSIS FUNCTION
                frame_to_analyze = st.session_state['captured_frame']
                analyze_image(frame_to_analyze)
            else:
                st.info("Go to 'Tab 2' and click 'Capture This Frame' first.")

# -------------------------------------------
# MODE B: SIMPLE IMAGE UPLOAD
# -------------------------------------------
elif app_mode == "Upload Image (Simple)":
    img_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        analyze_image(img)
