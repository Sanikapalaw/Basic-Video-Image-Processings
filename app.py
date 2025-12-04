import streamlit as st
import cv2
import numpy as np
import os
import time
import tempfile

# Function to check prime numbers for Q11
def is_prime(num):
    if num <= 1: return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0: return False
    return True

# Function to create dummy video if none is uploaded
def create_dummy_video(filename):
    width, height = 640, 480
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v for compatibility
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(30): # 30 frames
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Moving square
        cv2.rectangle(frame, (10 + i*10, 100), (110 + i*10, 200), (255, 255, 255), -1)
        # Static circle
        cv2.circle(frame, (400, 300), 50, (0, 0, 255), -1)
        out.write(frame)
    out.release()
    return filename

# ==========================================
# STREAMLIT UI START
# ==========================================

st.title("Practical 1: Image Processing")
st.write("Processing Video -> Extracting Image -> Solving Q1-Q11")

# --- Step 1: Input Video ---
st.header("1. Input Video")
option = st.radio("Choose Input Source:", ["Generate Dummy Video", "Upload Video"])

tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

if option == "Generate Dummy Video":
    if st.button("Generate Video"):
        create_dummy_video(tfile.name)
        st.success(f"Generated dummy video at {tfile.name}")
        st.video(tfile.name)
else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile.write(uploaded_file.read())
        st.success("Video Uploaded")
        st.video(tfile.name)

# --- Step 2: Extract Image ---
if os.path.getsize(tfile.name) > 0:
    st.header("2. Extract Frame (Q1)")
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        frame_idx = st.slider("Select Frame to Cut", 0, total_frames-1, 15)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # REMOVED use_container_width=True so image stays at natural size (640x480)
            st.image(frame_rgb, caption=f"Extracted Frame {frame_idx}")
            
            # Save extraction for processing
            img = frame # Keep in BGR for OpenCV processing steps below
        else:
            st.error("Could not read frame.")
            st.stop()
    else:
        st.warning("Please generate or upload a video first.")
        st.stop()
        
    cap.release()

    # --- Step 3: Practical Questions ---
    st.header("3. Practical Solutions (Q2 - Q11)")
    
    tabs = st.tabs(["Q2 Display", "Q3 B&W", "Q4 Props", "Q5 Rotate", "Q6 Mirror", "Q7 Object", "Q9/10 Cuts", "Q11 Grid"])

    # Q2: Show Image
    with tabs[0]:
        st.subheader("Q2: Display Image")
        st.image(frame_rgb, caption="Original Image (in.show)")

    # Q3: Black and White
    with tabs[1]:
        st.subheader("Q3: Black and White")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(gray_img, caption="Grayscale Image")

    # Q4: Properties
    with tabs[2]:
        st.subheader("Q4: Image Properties")
        h, w, c = img.shape
        file_stats = os.stat(tfile.name)
        size_mb = file_stats.st_size / (1024 * 1024)
        
        st.write(f"**Dimensions (Pixels):** {w} x {h}")
        st.write(f"**Channels:** {c}")
        st.write(f"**Video File Size:** {size_mb:.4f} MB")
        st.write(f"**Creation Date:** {time.ctime(file_stats.st_ctime)}")

    # Q5: Rotate
    with tabs[3]:
        st.subheader("Q5: Rotation")
        angle = st.radio("Select Angle:", ["90 Degrees", "180 Degrees", "270 Degrees"])
        
        if angle == "90 Degrees":
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == "180 Degrees":
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        st.image(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), caption=f"Rotated {angle}")

    # Q6: Mirror
    with tabs[4]:
        st.subheader("Q6: Mirror Image")
        mirror = cv2.flip(img, 1) # 1 = horizontal
        st.image(cv2.cvtColor(mirror, cv2.COLOR_BGR2RGB), caption="Mirrored Image")

    # Q7: Object Detection
    with tabs[5]:
        st.subheader("Q7: Object Detection (Thresholding)")
        thresh_val = st.slider("Threshold Value", 0, 255, 127)
        
        _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detect_img = img.copy()
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 500: # Filter noise
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                cv2.rectangle(detect_img, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)
                count += 1
        
        st.write(f"Objects Detected: {count}")
        st.image(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB), caption="Detected Objects (Green Box)")

    # Q9 & Q10: Cuts
    with tabs[6]:
        st.subheader("Q9 & Q10: Image Cuts")
        
        st.markdown("**Q9: Vertical Cut (80% - 20%)**")
        split_x = int(w * 0.8)
        col1, col2 = st.columns([8, 2])
        with col1:
            st.image(cv2.cvtColor(img[:, :split_x], cv2.COLOR_BGR2RGB), caption="Left 80%")
        with col2:
            st.image(cv2.cvtColor(img[:, split_x:], cv2.COLOR_BGR2RGB), caption="Right 20%")

        st.markdown("---")
        
        st.markdown("**Q10: Horizontal Cut (70% - 30%)**")
        split_y = int(h * 0.7)
        st.image(cv2.cvtColor(img[:split_y, :], cv2.COLOR_BGR2RGB), caption="Top 70%")
        st.image(cv2.cvtColor(img[split_y:, :], cv2.COLOR_BGR2RGB), caption="Bottom 30%")

    # Q11: Grid
    with tabs[7]:
        st.subheader("Q11: Grid (Prime Numbers Removed)")
        
        grid_img = img.copy()
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
                
                # Draw grid lines
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                # Optional: Add text to see numbers
                cv2.putText(grid_img, str(counter), (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                counter += 1
        
        st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), caption="Grid with Primes Removed")

else:
    st.info("Click 'Generate Video' or Upload a file to start.")
