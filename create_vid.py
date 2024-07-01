import cv2
import os

def frames_to_video(input_folder, output_video_path, frame_rate=5):
    # Get all the frame files in the input directory
    frames = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    frames.sort()  # Ensure the frames are in the correct order

    if not frames:
        print(f"No frames found in {input_folder}")
        return

    # Read the first frame to get the frame size
    first_frame_path = os.path.join(input_folder, frames[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    # Write each frame to the video
    for frame_file in frames:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video saved at {output_video_path}")

# Define the input folder and output video path
input_folder = 'result'
output_video_path = 'output_video.mp4'

# Combine frames into a video
frames_to_video(input_folder, output_video_path)