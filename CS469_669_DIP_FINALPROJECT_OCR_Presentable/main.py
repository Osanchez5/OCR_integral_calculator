import os
import cv2 as cv
import numpy as np
import pytesseract
import integral_math
from PIL import Image
from matplotlib import pyplot as plt

def zero_padding(image_array):
    padded_img = np.pad(image_array, 
                        pad_width=1, 
                        mode='constant', 
                        constant_values=0)
    return padded_img


def convert_to_grayscale(image_array):
    # Use weighted techinque to convert the image into grayscale
    grayscale_image = None

    if(len(image_array.shape) == 3):
        r = image_array[:,:,0]
        g = image_array[:,:,1]
        b = image_array[:,:,0]

        grayscale_image = (0.299 * r) + (0.587 * g) + (0.114 * b)
    else:
        grayscale_image = image_array
    return grayscale_image
import numpy as np


def otsu_thresh(image):
    binary_image = None
    # Flatten image and calculate histogram
    # Using numpys histogram function to create the histogram
    histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))
    height, width = image.shape
    total_pixels = height * width

    # Normalize the histogram
    probs = histogram / total_pixels

    # Class 0
    w0 = 0
    # Class 1
    w1 = 0

    # Initialize the means
    mu0 = 0
    mu1 = 0
    sum_for_mu0 = 0

    # Optimization trick
    # 255 bins
    bins = np.arange(256)
    mut = np.sum(bins * probs)
    # Define it now, start at 0
    best_between_class_var = 0

    # Find the optimal threshold
    optimal_threshold = 0

    # w0*w1 * ((mu0 - mu1) ** 2) 
    between_class_variance = 0

    for i in range(1, 255):
        # Probably of class 0 is fraction of pixels below T
        w0 += probs[i - 1]
        # Probablity of class 1 is the opposite of it
        w1 = 1 - w0
        # Get errors if this isn't here
        if(w0 == 0 or w1 == 0):
            continue

        # Average intensity of pixels in C0
        # From 0 to T - 1
        # The top values are summed and not the bottom values
        sum_for_mu0 += (i - 1) * probs[i - 1]
        mu0 = sum_for_mu0 / w0

        # mut = wo * mu0 + w1 * mu1
        # So mu1 = mut - w0 * mu0 / w1
        mu1 = (mut - (w0 * mu0)) / w1
        # Calculate the between class variance
        between_class_variance = w0 * w1 * ((mu0 - mu1)**2)

        # Determine the optimal threshold by comparing the current best class variance
        # With the recently computed between class variance and make the curr value the threshold
        if between_class_variance > best_between_class_var:
            best_between_class_var = between_class_variance
            optimal_threshold = i

    binary_image = image.copy()
    # Apply the threshold to the image through numpy.where
    binary_image = np.where(image >= optimal_threshold, 255, 0).astype(np.uint8)

    return binary_image

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
    for i in range(1, height):
        for j in range(1, width):
            # Do work in here
            curr_region = padded_img[i:i+se.shape[0], j:j+se.shape[1]]
            if np.any(curr_region * se):
                dilation_img[i, j] = 255
            else:
                dilation_img[i, j] = 0

    dilation_img = dilation_img * 255

    return dilation_img
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
    for i in range(1, height):
        for j in range(1, width):
            # Do work in here
            curr_region = padded_img[i:i+se.shape[0], j:j+se.shape[1]]
            if np.all(curr_region * se):
                erosion_img[i, j] = 255
            else:
                erosion_img[i, j] = 0

    erosion_img = erosion_img * 255
    return erosion_img

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

    gaussian_img = cv.filter2D(gaussian_img, -1, kernel)
    return gaussian_img

def read_in_image(directory):
    image = cv.imread(directory, flags = cv.IMREAD_COLOR_RGB)
    if image is None:
        print("Error reading in image")
        return

    # Do image preprocessing
    # Apply grayscale conversion
    gs_img = convert_to_grayscale(image)

    # Apply gaussian blur
    blurred_img = cv.GaussianBlur(gs_img, (7, 7), 1.5)
    # convert to a binary_img

    blurred_img = blurred_img.astype(np.uint8)
    binary_img = otsu_thresh(blurred_img)
    # binary_img = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary_img = binary_img.astype(np.uint8)

    kernel = np.array([1, 1, 1, 1, 1, 1, 1])
    dialted_image = cv.dilate(binary_img, kernel, iterations=1)

    # # Detect the edges
    # edges = canny_edge_detection(binary_img)
    # Output the changes to the image

    # Create a plotly showcase of everything done
    plt.figure(figsize = (15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.title("Original Image")

    plt.subplot(2, 3, 2)
    plt.imshow(gs_img, cmap = 'gray')
    plt.title("Grayscale applied")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(blurred_img, cmap = 'gray')
    plt.title("Blurred Image")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(binary_img, cmap = 'gray')
    plt.title("Otsu Thresholding")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(dialted_image, cmap = 'gray')
    plt.title("Dilated Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return dialted_image

def determine_gradient_class_version(image_array):
    # Initially relied on a different implementation, this one should be faster
    # Taken from lecture
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float32)
    sobel_y = np.array([[-1, -2, 1], [0, 0, 0], [1, 2, 1]], dtype = np.float32)
    # Initially calculated the gradients through the following formula
    # Gx = I(x+1, y) - I(x - 1, y), Gy = I(x, y + 1)-I(x, y-1)
    Gx = cv.filter2D(image_array, -1, sobel_x)
    Gy = cv.filter2D(image_array, -1, sobel_y)

    # Calculate the magnitude and orientation
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    gradient_orientation = np.arctan2(Gy, Gx) * 180 / np.pi

    return gradient_magnitude, gradient_orientation

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    # Blur the image
    blurred_image = image.copy()
    blurred_image = blurred_image.astype(np.float32)
    # Gradient Computation 
    g_mag, g_orient = determine_gradient_class_version(blurred_image)

    # Non-maximum suppression
    height, width = g_mag.shape
    suppressed = np.zeros_like(g_mag)
    g_orient[g_orient < 0] += 360
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 0
            r = 0
            if(0 <= g_orient[i, j] < 22.5) or (157.5 <= g_orient[i, j] < 202.5) or (337.5 <= g_orient[i,j] < 360):
                q = g_mag[i, j + 1]
                r = g_mag[i, j - 1]
            elif(22.5 <= g_orient[i, j] < 67.5) or (202.5 <= g_orient[i, j] < 247.5):
                q = g_mag[i + 1, j - 1]
                r = g_mag[i - 1, j + 1]
            elif(67.5 <= g_orient[i, j] < 112.5) or (247.5 <= g_orient[i, j] < 292.5):
                q = g_mag[i + 1, j]
                r = g_mag[i - 1, j]
            else:
                q = g_mag[i - 1, j - 1]
                r = g_mag[i + 1, j + 1]
            if g_mag[i, j] >= q and g_mag[i, j] >= r:
                suppressed[i, j] = g_mag[i, j]
    # Double thresholding
    max_magnitude = np.max(suppressed)
    high = high_threshold * max_magnitude / 255
    low = low_threshold * max_magnitude / 255

    strong_edges = (suppressed > high).astype(np.uint8) * 255
    weak_edges = ((suppressed >= low) & (suppressed <= high)).astype(np.uint8) * 255

    # Edge tracking by hysteresis
    edges = strong_edges.copy()
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if weak_edges[i, j] and np.any(strong_edges[i - 1: i + 2, j - 1:j + 2]):
                edges[i, j] = 255
    # Check the output and debug

    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap = 'gray')
    # plt.title("Original Image")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(edges, cmap = 'gray')
    # plt.title("Canny Edges")
    # plt.axis('off')

    # plt.tight_layout()

    # plt.show()
    # plt.savefig("Images/" + "canny_output.png")
    # plt.close()

    return edges

if __name__ == "__main__":
    image = read_in_image(directory="Images/test.jpg")
    if image is None:
        print("Could not read in the image")
        exit(1)

    # hardcoded to my machine, not sure what else to do
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    handwriting = pytesseract.image_to_string(image, lang = 'handwriting_nums', config='-c tessedit_char_whitelist=0123456789+-/x  --psm 6')

    if handwriting is None:
        print("Error reading handwriting")
        exit(0)

    print(handwriting)
    integral_math.determine_expression(handwriting)

    