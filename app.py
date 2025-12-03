import streamlit as st
import cv2
import numpy as np
import tempfile
from moviepy.editor import VideoFileClip
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Media Processing Lab", layout="wide", page_icon="🎬")

st.title("🎬 Multimedia Processing Lab")
st.markdown("""
**Welcome!** This app demonstrates basic processing for both **Images** and **Videos**.
Select your mode in the sidebar to begin.
""")

# --- SIDEBAR: MODE SELECTION ---
mode = st.sidebar.radio("Select Mode:", ["🖼️ Image Processing", "🎥 Video Processing"])
st.sidebar.markdown("---")

# ==========================================
# MODE 1: IMAGE PROCESSING
# ==========================================
if mode == "🖼️ Image Processing":
    st.header("🖼️ Image Playground")
    
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert file to OpenCV Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1) # BGR

        # Task Selection
        task = st.sidebar.selectbox("Choose Task:", [
            "View Original", 
            "Grayscale", 
            "Blur", 
            "Canny Edge Detection", 
            "Rotate / Flip"
        ])

        # --- LOGIC ---
        if task == "View Original":
            st.image(img, channels="BGR", caption="Original Image")
            
            # Show Properties
            h, w, c = img.shape
            st.info(f"Dimensions: {w}x{h} pixels | Channels: {c}")

        elif task == "Grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            col1, col2 = st.columns(2)
            col1.image(img, channels="BGR", caption="Original", use_container_width=True)
            col2.image(gray, caption="Grayscale", use_container_width=True)

        elif task == "Blur":
            k_size = st.sidebar.slider("Blur Intensity (Kernel)", 1, 50, 5)
            # Kernel size must be odd
            if k_size % 2 == 0: k_size += 1
            blurred = cv2.GaussianBlur(img, (k_size, k_size), 0)
            st.image(blurred, channels="BGR", caption=f"Blurred (Kernel: {k_size})")

        elif task == "Canny Edge Detection":
            t_lower = st.sidebar.slider("Lower Threshold", 0, 255, 50)
            t_upper = st.sidebar.slider("Upper Threshold", 0, 255, 150)
            edges = cv2.Canny(img, t_lower, t_upper)
            st.image(edges, caption="Edge Map")

        elif task == "Rotate / Flip":
            option = st.radio("Choose action:", ["Rotate 90° Clockwise", "Flip Horizontal"])
            if option == "Rotate 90° Clockwise":
                processed = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                processed = cv2.flip(img, 1)
            st.image(processed, channels="BGR")

    else:
        st.info("Upload an image from the sidebar to start.")

# ==========================================
# MODE 2: VIDEO PROCESSING
# ==========================================
elif mode == "🎥 Video Processing":
    st.header("🎥 Video Analysis & Filter Studio")

    uploaded_video = st.sidebar.file_uploader("Upload a Video (MP4/MOV)", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Save video to temp file (OpenCV needs a physical path)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # --- VIDEO INFO ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        # Display Stats
        st.sidebar.markdown("### 📊 Video Stats")
        st.sidebar.write(f"**Duration:** {duration:.2f} sec")
        st.sidebar.write(f"**FPS:** {fps:.2f}")
        st.sidebar.write(f"**Total Frames:** {frame_count}")

        col1, col2 = st.columns([1, 2])

        # --- PART A: AUDIO EXTRACTION ---
        with col1:
            st.subheader("🎵 Audio")
            if st.button("Extract Audio"):
                try:
                    clip = VideoFileClip(video_path)
                    if clip.audio:
                        audio_path = "extracted_audio.mp3"
                        clip.audio.write_audiofile(audio_path, logger=None)
                        st.success("Extracted!")
                        st.audio(audio_path)
                    else:
                        st.warning("No audio found.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # --- PART B: VISUAL PROCESSING ---
        with col2:
            st.subheader("🎞️ Visual Filters")
            
            filter_type = st.selectbox("Choose Filter", ["Original", "Grayscale", "Canny Edges", "Negative", "Sketch"])
            
            # Brightness Slider
            brightness = st.slider("Adjust Brightness", -100, 100, 0)
            
            if st.button("▶️ Play Processed Video"):
                st.caption("Processing live... (Resized for performance)")
                
                cap = cv2.VideoCapture(video_path)
                stframe = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 1. Resize (Crucial for web speed)
                    # We resize width to 640px, keeping aspect ratio
                    h, w = frame.shape[:2]
                    new_w = 640
                    new_h = int((new_w / w) * h)
                    frame = cv2.resize(frame, (new_w, new_h))

                    # 2. Brightness Adjustment
                    if brightness != 0:
                        frame = cv2.convertScaleAbs(frame, beta=brightness)

                    # 3. Apply Filters
                    if filter_type == "Grayscale":
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Back to 3 channels for display
                    
                    elif filter_type == "Canny Edges":
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 100, 200)
                        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                    elif filter_type == "Negative":
                        frame = cv2.bitwise_not(frame)
                    
                    elif filter_type == "Sketch":
                        # Basic sketch effect
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        inv = cv2.bitwise_not(gray)
                        blur = cv2.GaussianBlur(inv, (21, 21), 0)
                        inv_blur = cv2.bitwise_not(blur)
                        frame = cv2.divide(gray, inv_blur, scale=256.0)
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                    # 4. Display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                cap.release()
                st.success("Playback finished!")
    else:
        st.info("Upload a video to see stats and filters.")
