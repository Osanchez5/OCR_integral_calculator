import os
# import integral_math as inm # figure out how to import from the .py file
import cv2 as cv # Could try using pillow instead
import numpy as np
from matplotlib import pyplot as plt
# Import a machine learning library, perhaps one used in 422?

def zero_padding(image_array):
    padded_img = np.pad(image_array, 
                        pad_width=1, 
                        mode='constant', 
                        constant_values=0)
    return padded_img

# Convert an image to grayscale
def convert_to_grayscale(image_array):
    # Use weighted technique to convert the image into grayscale
    grayscale_image = None

    if(len(image_array.shape) == 3):
        # Assuming openCV is utilized
        r = image_array[:,:,2]
        g = image_array[:,:,1]
        b = image_array[:,:,0]
        
        grayscale_image = (0.299 * r) + (0.587 * g) + (0.114 * b)
    else:
        # The image is already grayscale so do nothing
        grayscale_image = image_array

    
    # plt.imshow(grayscale_image, cmap = 'gray')
    # plt.axis('off')
    # plt.title("Grayscale test")
    # plt.get_current_fig_manager().set_window_title("Grayscale Test")
    # plt.show()

    return grayscale_image

# Might not be needed after all
def convolution(image_array, kernel):
    filtered_image = image_array.copy()
    kernel_size = kernel.shape[0]
    swapped_kernel = np.transpose(np.transpose(kernel))

    # Come back to this
    for i in range(1, image_array.shape[0] - kernel_size + 1):
        for j in range(1, image_array.shape[1] - kernel_size + 1):
            curr_block = filtered_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(curr_block * swapped_kernel)

    return filtered_image

# This should be applied before determining gradients
def apply_gaussian(image_array, kernel_size, sigma):
    gaussian_img = image_array.copy()
    kernel = np.zeros((kernel_size, kernel_size))
    # A kernel needs to be applied to the entire image array?
    # Gaussian Kernels are defined as: w(s, t)
    # From the text book G(r) = Ke^(-(s^2+t^2)/2sigma^2)
    # Determine the value of the kernel

    center = np.floor(kernel_size / 2)
    s, t = np.indices((kernel_size, kernel_size))

    exponent = -((s - center)**2 + (t - center)**2) / (2 * sigma**2)
    # Essentially e^exponent
    gaussian = np.exp(exponent)

    kernel = gaussian / np.sum(gaussian)


    # Apply the filter
    # pad the image
    #padded_image = pad_image(gaussian_img, kernel_size)
    padded_image = zero_padding(gaussian_img)

    # Apply the filter
    gaussian_img = convolution(padded_image, kernel)

    gaussian_img = gaussian_img[1:gaussian_img.shape[0]-1, 1:gaussian_img.shape[1]-1]

    # plt.figure(figsize=(15,8))
    # plt.get_current_fig_manager().set_window_title("Gaussian Test")
    # plt.subplot(1, 2, 1)
    # plt.imshow(gaussian_img, cmap = 'gray')
    # plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(image_array, cmap = 'gray')
    # plt.axis('off')
    
    # plt.show()

    return gaussian_img

def pad_image(image_array, block_size):
    # Will be padded based on what the filter is applied
    width = image_array.shape[1]
    height = image_array.shape[0]

    # Refer to the pad_image function from assignment 2
    if(width % block_size == 0 and height % block_size == 0):
        return image_array
    
    padded_image = image_array.copy()

    # Should be grayscale image at this point, but still  account for RGB images
    if(len(image_array.shape) == 3):
        # Then the image is not grayscale
        # Pad for all channels
        padded_image = image_array.copy()
        # Pad the image
        # Loop until bother width and height are divisible by 8
        while(padded_image.shape[1] % block_size != 0):
            # From numpy documentation
            padded_image = np.pad(padded_image, [(0, 0), (0, 1), (0, 0)], mode = 'constant')
        while(padded_image.shape[0] % block_size != 0):
            padded_image = np.pad(padded_image, [(0, 1), (0, 0), (0, 0)], mode = 'constant')
    else:
        # The image is gray scale
        # Pad
        padded_image = image_array.copy()
        while(padded_image.shape[1] % block_size != 0):
            # From numpy documentation
            padded_image = np.pad(padded_image, [(0, 0), (0, 1)], mode = 'constant')
        while(padded_image.shape[0] % block_size != 0):
            padded_image = np.pad(padded_image, [(0, 1), (0, 0)], mode = 'constant')

    return np.uint8(padded_image)

def determine_gradient(image_array):
    # Taken from lecture slides
    # Gx = I(x+1, y) - I(x - 1, y), Gy = I(x, y + 1)-I(x, y-1)

    # Kept getting overflow errors so attempt to change the data type
    image_array = image_array.astype(np.float32)

    Gx = np.zeros_like(image_array)
    Gy = np.zeros_like(image_array)
    for i in range(1, image_array.shape[0] -1):
        for j in range(1, image_array.shape[1] -1):
            Gx[i, j] = image_array[i, j + 1] - image_array[i, j - 1]
            Gy[i, j] = image_array[i + 1, j] - image_array[i - 1, j]


    gradient_magnitude = np.sqrt((Gx*Gx) + (Gy*Gy))
    gradient_orientation = np.arctan2(Gy, Gx)

    
    # plt.figure(figsize=(10,5))
    # plt.get_current_fig_manager().set_window_title("Gradient Test")
    # plt.subplot(1, 3, 1)
    # plt.imshow(gradient_magnitude, cmap = 'gray')
    # plt.axis('off')
    
    # plt.subplot(1, 3, 2)
    # plt.imshow(image_array, cmap = 'gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(gradient_orientation, cmap = 'gray')
    # plt.axis('off')

    # plt.show()

    return gradient_magnitude, gradient_orientation

# Threshold the image to ensure HOG and Segmentation(?) work properly
def threshold_image(image_array, neighborhood_size, constant):
    # Try using local threshold

    threshold_img = np.zeros_like(image_array)
    
    # Try using local thresholding
    # Local thresholding requires either calculating the mean or median of a neighborhood to get it's threshold value
    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            neighborhood = image_array[i:i+neighborhood_size, j:j+neighborhood_size]
            neighborhood_mean = np.uint8(np.mean(neighborhood) - constant) # Might have to subtract by a constant

            if image_array[i, j] > neighborhood_mean:
                threshold_img[i, j] = 255 # Assuming 8 bit depth
            else:
                threshold_img[i, j] = 0

    # plt.imshow(threshold_img, cmap = 'gray')
    # plt.axis('off')
    # plt.title("Threshold Test")
    # plt.get_current_fig_manager().set_window_title("Threshold Test")
    
    
    # plt.show()

    return threshold_img

# dilation
def dilation(image_array):
    dilation_img = np.zeros_like(image_array)

    width = image_array.shape[1]
    height = image_array.shape[0]


    # Try this shape first?
    se = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    
    # If needed just pad the image the same way as above
    # Simply adds a border of zeros around the image_array
    padded_img = zero_padding(image_array)
    
    padded_img = padded_img // 255

    # Apply the structured element
    # Stride of 1
    for i in range(height):
        for j in range(width):
            # Do work in here
            curr_region = padded_img[i:i+se.shape[0], j:j+se.shape[1]]
            if np.any(curr_region * se):
                dilation_img[i, j] = 255
            else:
                dilation_img[i][j] = 0

    dilation_img = dilation_img * 255

    return dilation_img
# Erosion
def erosion(image_array):
    erosion_img = np.zeros_like(image_array)

    width = image_array.shape[1]
    height = image_array.shape[0]

    # Try this shape first?
    se = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    
    # If needed just pad the image the same way as above
    # Simply adds a border of zeros around the image_array
    padded_img = zero_padding(image_array)
    
    padded_img = padded_img // 255


    # Apply the structured element
    # Stride of 1
    for i in range(height):
        for j in range(width):
            # Do work in here
            curr_region = padded_img[i:i+se.shape[0], j:j+se.shape[1]]
            if np.all(curr_region * se):
                erosion_img[i, j] = 255
            else:
                erosion_img[i][j] = 0

    erosion_img = erosion_img * 255
    return erosion_img

def cell_division(image_array):
    cell_size = 4 # 4x4

    # Pad the image array
    padded_image = pad_image(image_array, cell_size)

    width = padded_image.shape[1]
    height = padded_image.shape[0]
    cells = []

    
    # Try overlapping the cells, should enhance the robustness of the descriptor
    for i in range(0, height - cell_size + 1):
        for j in range(0, width - cell_size + 1):
            cell = padded_image[i:i+cell_size, j:j+cell_size]
            cells.append(np.uint8(cell))

    # The cells should now be separated into a list and overlap each other
    return cells

def create_histogram(mag_cell, orient_cell, num_containers=9):
    num_containers = 9
    container_width = 20 # -> Should be 20
    cell_size = mag_cell.shape[0]
    cell_histogram = np.zeros(num_containers)
    # Loop through the image cells
    # From 0-180 <-> 9 containers
    
    # Each cell generates a histogram of gradient orientations, typically with 9 bins 
    # Bilinear interpolation distributes graident magnitudes across adjacent bins
    # I.e. bins next to each other
    # Used method 4 from https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/

    for i in range(cell_size - 1):
        for j in range(cell_size - 1):
            # Get the current magnitude and orientation in the cell
            curr_mag = mag_cell[i, j]
            curr_orient = orient_cell[i, j]
            # Bilinear Interpolation, determine the current and next bins to distribute
            # Magnitudes across adjacent bins
            curr_bin = int(curr_orient // container_width) % num_containers
            next_bin = (curr_bin + 1) % num_containers

            # ratio is to determine where the current magnitude should go based
            # upon the value of the current orientation
            ratio = (curr_orient % container_width) / container_width

            cell_histogram[curr_bin] += curr_mag * (1 - ratio)
            cell_histogram[next_bin] += curr_mag * ratio

    return cell_histogram
# A value is needed for epsilon
def block_normalization(histogram_created, epsilon = 1e-6):
    # Cells are grouped into blocks and their histograms are concatenated and normalized
    # Ensures the effects of illumination and contrast variations are mitigated
    cell_size = 2


    norm = np.linalg.norm(histogram_created)
    normalized_block = (histogram_created/np.sqrt(norm + epsilon))


    return normalized_block

def preprocess_image(image_array):
    
    # The image should already be in gray scale
    blurred_image = apply_gaussian(image_array, 5, 1.5)
    # Threshold the image
    thresholded_image = threshold_image(blurred_image, 21, 10)

    processed_image = thresholded_image
    # resize and crop it
    processed_image = resize_and_crop(processed_image)

    plt.imshow(processed_image, cmap = 'gray')
    plt.axis('off')
    plt.title("Crop Test")
    plt.get_current_fig_manager().set_window_title("Crop Test")
    
    
    plt.show()
    return processed_image
# Resize the image to match the sizing of the datasets fed into the SVM model
def resize_and_crop(image_array):
    height = image_array.shape[0]
    width = image_array.shape[1]
    # 1:2 aspect ratio desired
    output_height = 64
    output_width = 128

    # Determine the crop height and width
    target_width = min(width, int(height * 1/2))
    target_height = int(target_width * 1/2)

    starting_width = width // 2 - target_width //2
    starting_height = height // 2 - target_height // 2 
    # Crop to the center 1:2 rectangle
    cropped_image = image_array[starting_height:starting_height + target_height, starting_width:starting_width+target_width]

    # Resuze to 128x64
    resized_image = cv.resize(cropped_image, (output_width, output_height), interpolation = cv.INTER_AREA)

    return resized_image

def HOG_Testing(image_array):
    if(image_array.shape != 1):
        image_array = convert_to_grayscale(image_array)
    feature_descriptor = []
    # Otherwise continue
        # 2. Gradient Computation
    gradient_mag, gradient_orient = determine_gradient(image_array)

    # 3. Cell Division
    mag_cells = cell_division(gradient_mag)
    orient_cells = cell_division(gradient_orient)

    # Perhaps loop through the cells? to calculate each cells individually
    for i in range(len(mag_cells)):
        # 4. Histogram Creation
        histogram_created = np.array(create_histogram(mag_cells[i], orient_cells[i], 9))

        # 5. Block Normalization
        normalized_blocks = block_normalization(histogram_created)
        feature_descriptor.extend(normalized_blocks)

    

    return feature_descriptor

def histogram_of_oriented_gradients(image_array):
    # General process has been posted on canvas, follow that
    gs_img = image_array.copy()
    normalized_blocks = None
    feature_descriptor = []

    # 1. Grayscale conversion
    if gs_img.shape != 1:
        gs_img = convert_to_grayscale(image_array)
    print(gs_img.shape)
    gs_img = preprocess_image(gs_img)
    print(gs_img.shape)

    # 2. Gradient Computation
    gradient_mag, gradient_orient = determine_gradient(gs_img)

    # 3. Cell Division
    mag_cells = cell_division(gradient_mag)
    orient_cells = cell_division(gradient_orient)

    # Perhaps loop through the cells to calculate each cells individually
    for i in range(len(mag_cells)):
        # 4. Histogram Creation
        histogram_created = np.array(create_histogram(mag_cells[i], orient_cells[i], 9))

        # 5. Block Normalization
        normalized_blocks = block_normalization(histogram_created)
        feature_descriptor.extend(normalized_blocks)

    # 6. Descriptor Concatenation
    return feature_descriptor

def read_in_image(directory):

    feature_descriptor = None
    # Read in the image and save it to an image_array
    # Do appriopriate error checking

    # Once the image is read in apply the handwriting recognition process

    # All of the image processing is done expecting the image to use BGR
    image = cv.imread(directory, flags = cv.IMREAD_COLOR_BGR)

    if image is None:
        print("Error reading in image")
        return feature_descriptor

    
    # Resize the image?
    # Should return the feature descriptor
    feature_descriptor = handwriting_recognition_process(image)

    return feature_descriptor
# For testing with the SVM model
def pass_image(image_array):

    return HOG_Testing(image_array)

def handwriting_recognition_process(image_array):
    # Reduce image size
    # if image_array.shape[0] == image_array.shape[1]:
    #     image_array = cv.resize(image_array, (0, 0), fx=0.3, fy=0.3)


    gs_img = convert_to_grayscale(image_array)
    # Apply the image processing onto the grayscale image
    # gs_img = threshold_image(gs_img, 21, 10)

    # I will have to make the image smaller so that it only shows the handwriting detected?
    feature_descriptor = histogram_of_oriented_gradients(gs_img)

    # Not sure what to return
    # Figure out a way to write latex using python
    return feature_descriptor
