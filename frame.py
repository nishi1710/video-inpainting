import cv2
import os

def extract_frames(video_path, output_dir, fps=25):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the original video's frame rate
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval between frames to extract
    if original_fps < fps:
        frame_interval = 1  # Extract every frame if original FPS is less than desired FPS
    else:
        frame_interval = int(original_fps / fps)
    
    frame_count = 0
    saved_frame_count = 0
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Save the frame if it's at the desired interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{saved_frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()
    print(f"Extracted {saved_frame_count} frames and saved to {output_dir}")

# Example usage
video_path = 'video3s.mp4'
output_dir = 'frames'
extract_frames(video_path, output_dir, fps=25)
