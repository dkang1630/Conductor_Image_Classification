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

def calculate_sum_of_gray_values(image):
    rows, cols = image.shape
    row_sums = np.mean(image, axis=1)
    medfilt_row_sums = medfilt(row_sums, kernel_size=5)
    col_sums = np.mean(image, axis=0)
    medfilt_col_sums = medfilt(col_sums, kernel_size=5)
    return medfilt_row_sums, medfilt_col_sums, row_sums, col_sums

def find_nonzero_ranges(arr):
    ranges = []
    start = None
    for idx, num in enumerate(arr):
        if num != 0:
            if start is None:
                start = idx  # Start of a new contiguous range
        elif start is not None:
            ranges.append((start, idx - 1))  # End of the current contiguous range
            start = None  # Reset start for the next range
    if start is not None:
        ranges.append((start, len(arr) - 1))  # End of the last contiguous range
    return ranges

# Define a function to combine adjacent ranges
def combine_ranges(ranges, max_gap=5):
    combined_ranges = []
    start, end = ranges[0]
    for r in ranges[1:]:
        if r[0] <= end + max_gap:
            end = r[1]
        else:
            combined_ranges.append((start, end))
            start, end = r
    combined_ranges.append((start, end))
    return combined_ranges

def otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged
#folder path
image_folder_path = r'"C:\Users\dkang\OneDrive\Documents\Conductor_research\Image\test_image.jpg"'
# GVN_image_folder_path = r'C:\Users\dkang\OneDrive\Documents\Gray Scale Image Code\Conductor_research\Image\GVN\GVN_Img\Corrosion'
# GVN_Hist_Img_folder_path = r'C:\Users\dkang\OneDrive\Documents\Gray Scale Image Code\Conductor_research\Image\GVN\GVN_Hist_Img\Corrosion'
# GVN_Hist_data__row_folder_path = r'C:\Users\dkang\OneDrive\Documents\Gray Scale Image Code\Conductor_research\data\GVN\Corrosion\row'
# GVN_Hist_data__col_folder_path = r'C:\Users\dkang\OneDrive\Documents\Gray Scale Image Code\Conductor_research\data\GVN\Corrosion\col'

for filename in os.listdir(image_folder_path):
    
    image_path = os.path.join(image_folder_path, filename)
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    
    if gray_image is not None:
        
        normalized_map = gvn_processing(gray_image)
        normalized_map_uint16 = cv2.normalize(normalized_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint16)
        
        ##Save the plot with image
        # GVN_image_output_path = os.path.join(GVN_image_folder_path, f'GVN_{filename}.png')
        #cv2.imwrite(GVN_image_output_path, normalized_map_uint16)
        plt.figure()
        plt.imshow(normalized_map_uint16, cmap='gray')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.colorbar()
        plt.show()
        # plt.savefig(GVN_image_output_path)  # Save the plot as 'plot.png' in the output folder
        #plt.close()  # Close the current figure

        #Calculate the summation of gray values for each row and column
        medfilt_row_average, medfilt_col_average, row_average, col_average = calculate_sum_of_gray_values(normalized_map_uint16)
        # Save row average
        #np.savetxt(os.path.join(GVN_Hist_data__row_folder_path, f'row_average_{filename}_data.txt'), row_average)
        # Save column averages
        #np.savetxt(os.path.join(GVN_Hist_data__col_folder_path, f'col_average_{filename}_data.txt'), col_average)

        # Calculate the threshold T
        T_row = (np.max(medfilt_row_average) + np.min(medfilt_row_average)) / 2
        # Threshold the column sums based on T
        thresholded_row_average = np.where(medfilt_row_average > T_row, medfilt_row_average - T_row, 0) 
        T_col = (np.max(medfilt_col_average) + np.min(medfilt_col_average)) / 2
        # Threshold the column sums based on T
        thresholded_col_average = np.where(medfilt_col_average > T_col, medfilt_col_average - T_col, 0) 

        # Plot and save histograms for row average
        plt.figure(figsize=(10, 5))
        plt.plot(row_average, label='Original Row Sums', color='blue')
        plt.plot(medfilt_row_average, label='Median-Filtered Row Sums', color='red')
        plt.plot(thresholded_row_average, label='thresholded_row_average', color='green')
        plt.xlabel('Row Index (pixel)')
        plt.ylabel('Average of Gray Values (0-255)')
        plt.grid(True)
        plt.show()
        # plt.savefig(os.path.join(GVN_Hist_Img_folder_path, f'GVN_hist_row_{filename}.png'))  # Save the plot as plot1.png
        #plt.close()  # Close the current figure


        # Plot and save histograms for thresholded column average
        plt.figure(figsize=(10, 5))
        plt.plot(col_average, label='Original Column Sums', color='green')
        plt.plot(thresholded_col_average, label='Original Column Sums', color='red')
        plt.xlabel('Column Index (pixel)')
        plt.ylabel('Average of Gray Values (0 - 255)')
        plt.grid(True)
        plt.show()
        # plt.savefig(os.path.join(GVN_Hist_Img_folder_path, f'GVN_hist_col_{filename}.png'))  # Save the plot as plot1.png
        #plt.close()  # Close the current figure

        # Find ranges where thresholded_row_average is non-zero
        row_ranges = find_nonzero_ranges(thresholded_row_average)
        print("row_Range:",row_ranges)
        # Find ranges where thresholded_col_average is non-zero
        col_ranges = find_nonzero_ranges(thresholded_col_average)
        print(f"col_Range:",col_ranges)
       
       # Combine adjacent row and col ranges
        combined_row_ranges = combine_ranges(row_ranges)
        combined_col_ranges = combine_ranges(col_ranges)

       # Plot the image
        plt.imshow(normalized_map_uint16, cmap='gray')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
    
        # Draw rectangles based on the ranges
        for start_row, end_row in combined_row_ranges:
            for start_col, end_col in combined_col_ranges:
                rect_width = end_col - start_col + 1  # Width of the rectangle
                rect_height = end_row - start_row + 1  # Height of the rectangle
                rect = plt.Rectangle((start_col - 0.5, start_row - 0.5), rect_width, rect_height,
                                    edgecolor='red', linewidth=1, fill=False)
                plt.gca().add_patch(rect)
                # Crop the region of interest from the original image
                roi = normalized_map_uint16[start_row:end_row + 1, start_col:end_col + 1]

                # Apply edge detection (replace otsu_canny with your actual edge detection function)
                roi_uint8 = cv2.convertScaleAbs(roi)
                edged_roi = otsu_canny(roi_uint8)

                # Replace the corresponding region in the original image with the edge-detected version
                normalized_map_uint16[start_row:end_row + 1, start_col:end_col + 1] = edged_roi
        plt.imshow(normalized_map_uint16, cmap='gray')
        plt.show()


     


