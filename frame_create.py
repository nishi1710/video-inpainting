import cv2
import os

# Function to create a directory if it doesn't exist
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to extract frames from video
def extract_frames(video_path, output_folder, frames_to_extract=125, frame_rate=5):
    # Create output directory if it doesn't exist
    create_dir(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Get the original frame rate of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval between frames to capture
    interval = int(original_fps / frame_rate)
    
    # Initialize frame count and extracted frame count
    frame_count = 0
    extracted_frame_count = 0
    
    while cap.isOpened() and extracted_frame_count < frames_to_extract:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Check if the current frame is to be captured
        if frame_count % interval == 0:
            # Save the frame as a JPEG file
            frame_filename = os.path.join(output_folder, f'frame_{extracted_frame_count:03d}.jpg')
            cv2.imwrite(frame_filename, frame)
            extracted_frame_count += 1
        
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {extracted_frame_count} frames to {output_folder}")

# Define input video path and output folder
video_path = 'car_video_lesstime.mp4'
output_folder = 'frame_ouputs'

# Extract frames
extract_frames(video_path, output_folder)
