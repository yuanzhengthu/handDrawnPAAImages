import cv2
import os
import tifffile
import numpy as np
import PIL.Image as Image


def denoise_image(image_path, output_directory):
    # Load the image
    image = cv2.imread(image_path)

    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)

    # Convert the denoised image to grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # Create an adaptive histogram equalization object
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    # Apply adaptive histogram equalization
    clahe_equalized = clahe.apply(gray)

    # Create the output filename
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_directory, "denoised_" + filename)

    # Save the denoised image
    cv2.imwrite(output_path, clahe_equalized)


def adjust_pixel_threshold(input_tiff_path, output_directory, threshold=48):
    # Load the multi-page TIFF image
    image = cv2.imread(input_tiff_path)

    # Clip pixel values to the specified threshold
    modified_image = np.clip(image, threshold, 255).astype('uint8') - 48

    # Create the output filename
    filename = os.path.basename(input_tiff_path)
    output_path = os.path.join(output_directory, "denoised_" + filename)

    # Save the modified image
    cv2.imwrite(output_path, modified_image)


# Directory containing the images
input_directory = "dataset/input"

# Output directory to save the denoised and overlaid images
output_directory = "dataset/output"

# Recursively loop through all the image files in the main directory and its subdirectories
for root, _, files in os.walk(input_directory):
    for filename in files:
        if filename.endswith(".tif") or filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            adjust_pixel_threshold(image_path, output_directory)

# Iterate over each file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct the full file path
        image_path = os.path.join(input_directory, filename)

        # Call the denoise_image function
        denoise_image(image_path, output_directory)
