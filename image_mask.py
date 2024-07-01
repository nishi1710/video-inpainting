import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Class IDs for vehicles in the COCO dataset
vehicle_class_ids = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Predict with the model on the image
image_path = 'bus.jpg'
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
mask_image_path = 'vehicle_mask.png'
cv2.imwrite(mask_image_path, mask)

# Display the mask
# cv2.imshow('Vehicle Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load the image and the mask
image = cv2.imread('bus.jpg')
mask = cv2.imread('vehicle_mask.png')

# Check if the image and the mask are loaded correctly
if image is None or mask is None:
    raise ValueError("Could not open or find the image or mask")

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

# Display the original image, mask, grayscale result, and color result
cv2.imshow('Original Image', image)
#cv2.imshow('gray image' , gray_image)
cv2.imshow('Mask', gray_mask)
cv2.imshow('Grayscale Result', result)
cv2.imshow('Result in Color', result_color)
cv2.imshow('Applied Mask to Color Image', color_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
