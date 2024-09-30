import cv2
import numpy as np
from skimage.segmentation import chan_vese
from plantcv import plantcv as pcv
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt


def calculate_mean_gray_value(template):
    return np.mean(template)

def calculate_variance(pixel_grayVal, mean_gray_value):
    return np.mean((pixel_grayVal - mean_gray_value) ** 2)

def normalize_gray_scale_variance(gray_scale_variance_map):
    qmin = np.min(gray_scale_variance_map)
    qmax = np.max(gray_scale_variance_map)
    return (gray_scale_variance_map - qmin) / (qmax - qmin)

def gvn_processing(gray_image):
    rows, cols = gray_image.shape
    gray_scale_variance_map = np.zeros((rows, cols), dtype=np.float32)
    template_size = 3
    
    # Iterate over the image with overlapping cells
    for i in range(1, rows - template_size):
        for j in range(1, cols - template_size):

            template = gray_image[i-1:i+(template_size-1), j-1:j+(template_size-1)]
            mean_gray_value = calculate_mean_gray_value(template)
            gray_scale_variance = calculate_variance(template, mean_gray_value)
            gray_scale_variance_map[i, j] = gray_scale_variance
    
    normalized_map = normalize_gray_scale_variance(gray_scale_variance_map)
    
    return normalized_map


#folder path
image_folder_path = r"C:\Users\dkang\OneDrive\Documents\Conductor_research\Image\test"
GVN_image_folder_path = r'C:\Users\dkang\OneDrive\Documents\Conductor_research\Image'


for filename in os.listdir(image_folder_path):
    
    image_path = os.path.join(image_folder_path, filename)
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if gray_image is not None:
        
        normalized_map = gvn_processing(gray_image)
        normalized_map_uint8 = cv2.normalize(normalized_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        #Save the plot with image
        GVN_image_output_path = os.path.join(GVN_image_folder_path, f'GVN_{filename}.png')
        cv2.imwrite(GVN_image_output_path, normalized_map_uint8)
        # plt.imshow(normalized_map_uint8, cmap='gray')
        # plt.savefig(GVN_image_output_path)  # Save the plot as 'plot.png' in the output folder
        # plt.close()  # Close the current figure
   
