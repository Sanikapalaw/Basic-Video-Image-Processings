import streamlit as st
import tempfile
import os
# Import moviepy.editor to handle video and audio
from moviepy.editor import VideoFileClip

st.title("🎵 Video to Audio Separator")
st.markdown("Upload a video file to extract the audio automatically.")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Extract Audio"):
        with st.spinner("Separating audio..."):
            try:
                # 2. Save uploaded file to a temporary file
                # MoviePy requires a file path, it cannot read raw bytes directly
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.flush() # Ensure data is written
                tfile.close() # Close file so other processes can read it

                # 3. Load the video clip
                video_clip = VideoFileClip(tfile.name)
                
                # 4. Extract the audio
                audio_path = "extracted_audio.mp3"
                video_clip.audio.write_audiofile(audio_path)

                # 5. Display Audio Player
                st.success("Audio extracted successfully!")
                st.audio(audio_path)

                # 6. Download Button
                with open(audio_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Audio (.mp3)",
                        data=f,
                        file_name="audio.mp3",
                        mime="audio/mpeg"
                    )

                # Cleanup: Close clips and remove temp file
                video_clip.close()
                os.remove(tfile.name)

            except Exception as e:
                st.error(f"An error occurred: {e}")
