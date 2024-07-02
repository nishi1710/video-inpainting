import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Class IDs for vehicles in the COCO dataset
vehicle_class_ids = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

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

def process_image(image_path, output_mask_dir, output_result_dir):
    # Predict with the model on the image
    results = model(image_path)[0]

    # Load the image
    image = cv2.imread(image_path)
    H, W, _ = image.shape

    # Initialize an empty mask
    mask = np.zeros((H, W), dtype=np.uint8)

    # Process the segmentation results to create the mask for vehicles only
    for i, box in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = box

        if score > 0.25 and class_id in vehicle_class_ids:  # Threshold for detection confidence and filter for vehicles
            mask_instance = results.masks.data[i].cpu().numpy()
            mask_instance = cv2.resize(mask_instance, (W, H))
            mask_instance = (mask_instance > 0.5).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, mask_instance)

    # Save the mask image
    mask_image_path = os.path.join(output_mask_dir, os.path.basename(image_path))
    cv2.imwrite(mask_image_path, mask)

    # Load the image and the mask
    mask = cv2.imread(mask_image_path)

    # Resize the mask to match the image size (if necessary)
    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Convert to grayscale if they are not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if len(mask.shape) == 3:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = mask

    # Ensure the mask is in the correct range (0-255)
    gray_mask = cv2.normalize(gray_mask, None, 0, 255, cv2.NORM_MINMAX)

    # Subtract the mask from the image
    result = cv2.subtract(gray_image, gray_mask)

    # Convert the single-channel result back to a 3-channel image
    result_color = cv2.merge([result, result, result])

    # Alternatively, apply the grayscale result as a mask to the original color image
    # Create a mask to apply on the original color image
    mask_3_channel = cv2.merge([gray_mask, gray_mask, gray_mask])
    color_result = cv2.subtract(image, mask_3_channel)

    # Save the result images
    result_image_path = os.path.join(output_result_dir, os.path.basename(image_path))
    cv2.imwrite(result_image_path, color_result)

def process_directory(input_dir, output_mask_dir, output_result_dir):
    # Create output directories if they don't exist
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_result_dir, exist_ok=True)

    # Process each image in the input directory
    for image_filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_filename)
        if os.path.isfile(image_path):
            process_image(image_path, output_mask_dir, output_result_dir)

def frames_to_video(input_folder, output_video_path, frame_rate=25):
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

# Example usage
video_path = 'video3s.mp4'
frames_output_dir = 'frame_Com'
mask_output_dir = 'mask_Com'
result_output_dir = 'result_Com'
output_video_path = 'output_video.mp4'

# Step 1: Extract frames from the video
extract_frames(video_path, frames_output_dir, fps=25)

# Step 2: Process the extracted frames
process_directory(frames_output_dir, mask_output_dir, result_output_dir)

# Step 3: Combine processed frames into a video
frames_to_video(result_output_dir, output_video_path, frame_rate=25)
