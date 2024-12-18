import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

def crop_images(search_path, label_path):
    files = [img for img in glob.glob(search_path, recursive=True)]
    print(files)
    
    images = [np.asarray(Image.open(img)) for img in files]
    
    files_labels = [img for img in glob.glob(label_path, recursive=True)]
    labels = [np.asarray(Image.open(img)) for img in files_labels]

    # Create binary masks (1 where image > 0)
    masks = [(img > 0).astype('f4') for img in images]
    
    # Final mask is the sum of all masks, clipped between 0 and 1
    final_mask = np.clip(np.sum(np.array(masks, dtype='f4'), axis=0), 0, 1)
    
    # Preprocess the mask to make it stronger
    final_mask = (final_mask * 255).astype("u1")  # Convert to uint8
    
    # Apply Gaussian blur to smooth the mask
    blurred_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
    
    # Apply a binary threshold to make it more binary
    _, binary_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Perform morphological operations to remove small noise and close gaps
    kernel_1 = np.ones((15, 15), np.uint8)
    #kernel_2 = np.ones((15, 15), np.uint8)
    #cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.erode(binary_mask, kernel_1, iterations=1)
    #cleaned_mask = cv2.dilate(cleaned_mask, kernel_2, iterations=4)
    
    # Ensure the mask is a single-channel 8-bit image (required for cv2.findContours)
    if len(cleaned_mask.shape) == 3:
        cleaned_mask = cv2.cvtColor(cleaned_mask, cv2.COLOR_BGR2GRAY)
    
    cleaned_mask = cleaned_mask.astype(np.uint8)

    plt.imshow(cleaned_mask, cmap='gray')
    plt.title("Cleaned Binary Mask")
    plt.show()

    # Detect contours
    cnts, _ = cv2.findContours(
        cleaned_mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours on a copy of the cleaned mask
    cleaned_mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    mask_with_contours = cleaned_mask_bgr.copy()
    cv2.drawContours(mask_with_contours, cnts, -1, (0, 255, 0), 3)
    
    # Display the image with contours drawn
    plt.imshow(mask_with_contours)
    plt.title("Contours on Mask")
    plt.show()

    # Find the largest contour after filtering
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(c)

        # Crop the images based on the bounding rectangle
        cropped = [img[y: y + h, x: x + w] for img in images]
        cropped_labels = [img[y: y + h, x: x + w] for img in labels]

        # Save the cropped images
        save_path = "C:/Users/carla/Documents/Master_Thesis/Data/DataNewCrop/Frames/video_8/"
        save_path_labels = "C:/Users/carla/Documents/Master_Thesis/Data/DataNewCrop/Masks/video_8/"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, (img, f) in enumerate(zip(cropped, files)):
            cv2.imwrite(os.path.join(save_path, f"frame_{i:05d}.tif"), img)

        for i, (img, f) in enumerate(zip(cropped_labels,files_labels)):
            cv2.imwrite(os.path.join(save_path_labels, f"frame_{i:05d}.tif"), img)

# Use the function
video_path = "C:/Users/carla/Documents/Master_Thesis/Data/DataNew/Frames/video_8/*.tif"
label_path = "C:/Users/carla/Documents/Master_Thesis/Data/DataNew/Masks/video_8/*.tif"
crop_images(video_path, label_path)