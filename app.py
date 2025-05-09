import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

"""
Advanced Image Processing Laboratory
-----------------------------------
This Streamlit application provides a comprehensive suite of image processing tools
using OpenCV. Users can upload images and apply various transformations and 
filters to process and analyze images.

Main features:
- Basic operations (resize, rotate, flip, brightness/contrast)
- Filtering (blur, gaussian, median, bilateral)
- Color space transformations
- Thresholding techniques
- Morphological operations
- Edge and feature detection
- Histogram analysis and equalization
- Advanced visual effects

Author: Project 02
"""

def apply_threshold(image, threshold_type, thresh_value=127, max_value=255):
    """
    Apply various thresholding operations to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image
    threshold_type : str
        Type of thresholding to apply ('Binary', 'Binary Inverse', etc.)
    thresh_value : int
        Threshold value (0-255)
    max_value : int
        Maximum value to use with the threshold
        
    Returns:
    --------
    numpy.ndarray
        Thresholded image
    """
    if threshold_type == "Binary":
        _, thresh = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_BINARY)
    elif threshold_type == "Binary Inverse":
        _, thresh = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_BINARY_INV)
    elif threshold_type == "Truncate":
        _, thresh = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_TRUNC)
    elif threshold_type == "To Zero":
        _, thresh = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_TOZERO)
    elif threshold_type == "To Zero Inverse":
        _, thresh = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_TOZERO_INV)
    elif threshold_type == "Adaptive Mean":
        thresh = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    elif threshold_type == "Adaptive Gaussian":
        thresh = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif threshold_type == "Otsu":
        _, thresh = cv2.threshold(image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def apply_histogram_equalization(image, method):
    """
    Apply histogram equalization to enhance image contrast.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale or color)
    method : str
        Equalization method ('Simple' or 'CLAHE')
        
    Returns:
    --------
    numpy.ndarray
        Equalized image
    """
    if method == "Simple":
        if len(image.shape) == 3:
            # For color images, convert to YUV and equalize Y channel
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            # For grayscale images
            return cv2.equalizeHist(image)
    elif method == "CLAHE":
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(image.shape) == 3:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            return clahe.apply(image)
    return image

def apply_color_quantization(image, k):
    """
    Reduce the number of colors in an image using K-means clustering.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input color image
    k : int
        Number of colors to quantize to
        
    Returns:
    --------
    numpy.ndarray
        Color quantized image
    """
    # Reshape the image to a 2D array of pixels
    data = np.float32(image).reshape((-1,3))
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values and reshape to original image dimensions
    center = np.uint8(center)
    result = center[label.flatten()]
    return result.reshape(image.shape)

def main():
    """
    Main application function that sets up the Streamlit interface and
    handles all image processing operations.
    """
    # Configure the Streamlit page
    st.set_page_config(layout="wide")
    st.title("Advanced Image Processing Laboratory")
    
    # Add custom CSS styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create the sidebar control panel
    st.sidebar.title("Controls Panel")
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Process the uploaded image if available
    if uploaded_file is not None:
        # Load the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Display original image
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

        # Main processing options selector
        processing_option = st.sidebar.selectbox(
            "Select Processing Category",
            ["Basic Operations", "Filtering", "Color Spaces", "Thresholding", 
             "Morphological Operations", "Edge Detection", "Feature Detection",
             "Histogram Operations", "Advanced Effects"]
        )

        # Process the image based on selected options
        with col2:
            st.subheader("Processed Image")
            
            # ======= BASIC OPERATIONS SECTION =======
            if processing_option == "Basic Operations":
                st.sidebar.markdown("""
                ### Basic Operations
                Transform your image with fundamental operations:
                - **Resize**: Scale the image up or down
                - **Rotate**: Rotate the image by any angle
                - **Flip**: Mirror the image horizontally or vertically
                - **Brightness/Contrast**: Adjust image lighting
                - **Color Quantization**: Reduce the number of colors
                """)
                
                operation = st.sidebar.selectbox(
                    "Select Operation",
                    ["Resize", "Rotate", "Flip", "Brightness/Contrast", "Color Quantization"]
                )
                
                if operation == "Resize":
                    st.sidebar.markdown("Adjust the scale factor to resize the image. Values > 1 enlarge, values < 1 shrink.")
                    scale = st.sidebar.slider("Scale Factor", 0.1, 2.0, 1.0)
                    processed_img = cv2.resize(original_img, None, fx=scale, fy=scale)
                
                elif operation == "Rotate":
                    st.sidebar.markdown("Rotate the image by specifying an angle in degrees. Positive values rotate counter-clockwise.")
                    angle = st.sidebar.slider("Angle", -180, 180, 0)
                    center = (original_img.shape[1] // 2, original_img.shape[0] // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    processed_img = cv2.warpAffine(original_img, matrix, (original_img.shape[1], original_img.shape[0]))
                
                elif operation == "Flip":
                    st.sidebar.markdown("Mirror the image in different directions.")
                    flip_option = st.sidebar.selectbox("Flip Direction", ["Horizontal", "Vertical", "Both"])
                    if flip_option == "Horizontal":
                        processed_img = cv2.flip(original_img, 1)
                    elif flip_option == "Vertical":
                        processed_img = cv2.flip(original_img, 0)
                    else:
                        processed_img = cv2.flip(original_img, -1)
                
                elif operation == "Brightness/Contrast":
                    st.sidebar.markdown("""
                    Adjust image brightness and contrast:
                    - **Brightness**: Negative values darken, positive values brighten
                    - **Contrast**: Negative values decrease contrast, positive values increase it
                    """)
                    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
                    contrast = st.sidebar.slider("Contrast", -100, 100, 0)
                    
                    processed_img = original_img.copy()
                    # Apply brightness adjustment
                    if brightness != 0:
                        if brightness > 0:
                            shadow = brightness
                            highlight = 255
                        else:
                            shadow = 0
                            highlight = 255 + brightness
                        alpha_b = (highlight - shadow)/255
                        gamma_b = shadow
                        processed_img = cv2.addWeighted(processed_img, alpha_b, processed_img, 0, gamma_b)
                    
                    # Apply contrast adjustment
                    if contrast != 0:
                        f = 131*(contrast + 127)/(127*(131-contrast))
                        alpha_c = f
                        gamma_c = 127*(1-f)
                        processed_img = cv2.addWeighted(processed_img, alpha_c, processed_img, 0, gamma_c)

                elif operation == "Color Quantization":
                    st.sidebar.markdown("Reduce the number of colors in the image. Lower values create more poster-like effects.")
                    k = st.sidebar.slider("Number of Colors", 2, 16, 8)
                    processed_img = apply_color_quantization(original_img, k)

            # ======= FILTERING SECTION =======
            elif processing_option == "Filtering":
                st.sidebar.markdown("""
                ### Image Filtering
                Apply different filters to smooth or enhance the image:
                - **Blur**: Simple averaging filter
                - **Gaussian**: Weighted gaussian smoothing
                - **Median**: Good for removing salt-and-pepper noise
                - **Bilateral**: Edge-preserving smoothing
                - **Custom Kernel**: Apply specific filter effects
                """)
                
                filter_type = st.sidebar.selectbox(
                    "Select Filter",
                    ["Blur", "Gaussian", "Median", "Bilateral", "Custom Kernel"]
                )
                
                if filter_type == "Custom Kernel":
                    st.sidebar.markdown("Apply predefined kernel effects. Larger kernel sizes create stronger effects.")
                    kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
                    kernel_type = st.sidebar.selectbox("Kernel Type", ["Sharpen", "Edge Detection", "Emboss"])
                    
                    # Define custom convolution kernels
                    if kernel_type == "Sharpen":
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    elif kernel_type == "Edge Detection":
                        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
                    elif kernel_type == "Emboss":
                        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
                    
                    # Apply the convolution
                    processed_img = cv2.filter2D(original_img, -1, kernel)
                else:
                    st.sidebar.markdown("Adjust kernel size to control the strength of the filter effect.")
                    kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
                    
                    if filter_type == "Bilateral":
                        st.sidebar.markdown("""
                        Bilateral Filter Parameters:
                        - **d**: Diameter of pixel neighborhood
                        - **Sigma Color**: Filter sigma in color space
                        - **Sigma Space**: Filter sigma in coordinate space
                        """)
                        d = st.sidebar.slider("d", 1, 15, 9)
                        sigma_color = st.sidebar.slider("Sigma Color", 1, 255, 75)
                        sigma_space = st.sidebar.slider("Sigma Space", 1, 255, 75)
                        processed_img = cv2.bilateralFilter(original_img, d, sigma_color, sigma_space)
                    else:
                        # Apply standard filters
                        if filter_type == "Blur":
                            processed_img = cv2.blur(original_img, (kernel_size, kernel_size))
                        elif filter_type == "Gaussian":
                            processed_img = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), 0)
                        elif filter_type == "Median":
                            processed_img = cv2.medianBlur(original_img, kernel_size)

            # ======= COLOR SPACES SECTION =======
            elif processing_option == "Color Spaces":
                st.sidebar.markdown("""
                ### Color Spaces
                Convert the image between different color representations:
                - **RGB**: Standard Red-Green-Blue color space
                - **HSV**: Hue-Saturation-Value, useful for color segmentation
                - **LAB**: Perceptually uniform color space
                - **YCrCb**: Used in video encoding
                - **Individual Channels**: View color components separately
                """)
                
                color_space = st.sidebar.selectbox(
                    "Select Color Space",
                    ["RGB", "HSV", "LAB", "YCrCb", "Individual Channels"]
                )
                
                # Convert image to selected color space
                if color_space == "RGB":
                    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                elif color_space == "HSV":
                    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
                elif color_space == "LAB":
                    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
                elif color_space == "YCrCb":
                    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)
                elif color_space == "Individual Channels":
                    channel = st.sidebar.selectbox("Select Channel", ["Blue", "Green", "Red"])
                    # Extract the selected color channel
                    if channel == "Blue":
                        processed_img = original_img[:,:,0]
                    elif channel == "Green":
                        processed_img = original_img[:,:,1]
                    else:
                        processed_img = original_img[:,:,2]

            # ======= THRESHOLDING SECTION =======
            elif processing_option == "Thresholding":
                # Convert to grayscale for thresholding
                gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                threshold_type = st.sidebar.selectbox(
                    "Select Threshold Type",
                    ["Binary", "Binary Inverse", "Truncate", "To Zero", "To Zero Inverse",
                     "Adaptive Mean", "Adaptive Gaussian", "Otsu"]
                )
                
                # Apply selected thresholding method
                if threshold_type in ["Binary", "Binary Inverse", "Truncate", "To Zero", "To Zero Inverse"]:
                    thresh_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
                    max_value = st.sidebar.slider("Maximum Value", 0, 255, 255)
                    processed_img = apply_threshold(gray_img, threshold_type, thresh_value, max_value)
                else:
                    processed_img = apply_threshold(gray_img, threshold_type)

            # ======= MORPHOLOGICAL OPERATIONS SECTION =======
            elif processing_option == "Morphological Operations":
                operation = st.sidebar.selectbox(
                    "Select Operation",
                    ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]
                )
                
                # Create structuring element (kernel)
                kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                # Apply selected morphological operation
                if operation == "Erosion":
                    processed_img = cv2.erode(original_img, kernel, iterations=1)
                elif operation == "Dilation":
                    processed_img = cv2.dilate(original_img, kernel, iterations=1)
                elif operation == "Opening":
                    processed_img = cv2.morphologyEx(original_img, cv2.MORPH_OPEN, kernel)
                elif operation == "Closing":
                    processed_img = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel)
                elif operation == "Gradient":
                    processed_img = cv2.morphologyEx(original_img, cv2.MORPH_GRADIENT, kernel)
                elif operation == "Top Hat":
                    processed_img = cv2.morphologyEx(original_img, cv2.MORPH_TOPHAT, kernel)
                elif operation == "Black Hat":
                    processed_img = cv2.morphologyEx(original_img, cv2.MORPH_BLACKHAT, kernel)

            # ======= EDGE DETECTION SECTION =======
            elif processing_option == "Edge Detection":
                st.sidebar.markdown("""
                ### Edge Detection
                Different methods to detect edges in the image:
                - **Canny**: Advanced edge detector with thresholds
                - **Sobel**: Directional gradient detection
                - **Laplacian**: Detect edges using 2nd derivatives
                - **Scharr**: More accurate gradient calculation
                """)
                
                detector = st.sidebar.selectbox(
                    "Select Detector",
                    ["Canny", "Sobel", "Laplacian", "Scharr"]
                )
                
                # Apply selected edge detection method
                if detector == "Canny":
                    threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
                    threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
                    processed_img = cv2.Canny(original_img, threshold1, threshold2)
                
                elif detector == "Sobel":
                    dx = st.sidebar.slider("dx", 0, 2, 1)
                    dy = st.sidebar.slider("dy", 0, 2, 1)
                    ksize = st.sidebar.slider("Kernel Size", 1, 7, 3, step=2)
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    processed_img = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
                    processed_img = np.uint8(np.absolute(processed_img))
                
                elif detector == "Laplacian":
                    ksize = st.sidebar.slider("Kernel Size", 1, 7, 3, step=2)
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    processed_img = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                    processed_img = np.uint8(np.absolute(processed_img))
                
                elif detector == "Scharr":
                    direction = st.sidebar.selectbox("Direction", ["X", "Y"])
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    if direction == "X":
                        processed_img = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                    else:
                        processed_img = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                    processed_img = np.uint8(np.absolute(processed_img))

            # ======= FEATURE DETECTION SECTION =======
            elif processing_option == "Feature Detection":
                st.sidebar.markdown("""
                ### Feature Detection
                Detect interesting points or features in the image:
                - **Harris Corner**: Detects corner points using intensity changes
                - **Shi-Tomasi**: More robust corner detection
                - **FAST**: High-speed corner detection
                
                Parameters for Harris Corner:
                - **Block Size**: Size of neighborhood considered
                - **Kernel Size**: Aperture parameter for Sobel operator
                - **k**: Harris detector free parameter
                """)
                
                detector = st.sidebar.selectbox(
                    "Select Detector",
                    ["Harris Corner", "Shi-Tomasi", "FAST"]
                )
                
                # Apply selected feature detection algorithm
                if detector == "Harris Corner":
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    block_size = st.sidebar.slider("Block Size", 2, 10, 2)
                    ksize = st.sidebar.slider("Kernel Size", 3, 31, 3, step=2)
                    k = st.sidebar.slider("k", 0.01, 0.1, 0.04, step=0.01)
                    
                    processed_img = original_img.copy()
                    gray = np.float32(gray)
                    dst = cv2.cornerHarris(gray, block_size, ksize, k)
                    dst = cv2.dilate(dst, None)
                    # Mark corners with red color
                    processed_img[dst > 0.01 * dst.max()] = [0, 0, 255]
                
                elif detector == "Shi-Tomasi":
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
                    corners = np.int0(corners)
                    processed_img = original_img.copy()
                    # Mark corners with red circles
                    for i in corners:
                        x, y = i.ravel()
                        cv2.circle(processed_img, (x, y), 3, [0, 0, 255], -1)
                
                elif detector == "FAST":
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    # Create FAST feature detector object
                    fast = cv2.FastFeatureDetector_create()
                    kp = fast.detect(gray, None)
                    processed_img = original_img.copy()
                    # Draw keypoints
                    cv2.drawKeypoints(original_img, kp, processed_img, color=(0, 0, 255))

            # ======= HISTOGRAM OPERATIONS SECTION =======
            elif processing_option == "Histogram Operations":
                st.sidebar.markdown("""
                ### Histogram Operations
                Analyze and modify image intensity distribution:
                - **Show Histogram**: Display color/intensity distribution
                - **Equalization**: Enhance contrast using histogram equalization
                - **CLAHE**: Contrast Limited Adaptive Histogram Equalization
                """)
                
                operation = st.sidebar.selectbox(
                    "Select Operation",
                    ["Show Histogram", "Equalization", "CLAHE"]
                )
                
                if operation == "Show Histogram":
                    # Plot histogram of color channels or grayscale intensity
                    fig, ax = plt.subplots()
                    if len(original_img.shape) == 3:
                        colors = ('b', 'g', 'r')
                        for i, color in enumerate(colors):
                            hist = cv2.calcHist([original_img], [i], None, [256], [0, 256])
                            ax.plot(hist, color=color)
                    else:
                        hist = cv2.calcHist([original_img], [0], None, [256], [0, 256])
                        ax.plot(hist)
                    st.pyplot(fig)
                    processed_img = original_img
                
                elif operation == "Equalization":
                    processed_img = apply_histogram_equalization(original_img, "Simple")
                
                elif operation == "CLAHE":
                    processed_img = apply_histogram_equalization(original_img, "CLAHE")

            # ======= ADVANCED EFFECTS SECTION =======
            elif processing_option == "Advanced Effects":
                st.sidebar.markdown("""
                ### Advanced Effects
                Apply complex image transformations:
                - **Pencil Sketch**: Convert image to pencil drawing style
                - **Cartoon**: Create cartoon-like effect
                - **HDR Effect**: Enhance local details
                """)
                
                effect = st.sidebar.selectbox(
                    "Select Effect",
                    ["Pencil Sketch", "Cartoon", "HDR Effect"]
                )
                
                if effect == "Pencil Sketch":
                    # Convert to grayscale and invert
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    inv = 255 - gray
                    # Apply Gaussian blur and divide
                    blur = cv2.GaussianBlur(inv, (21, 21), 0)
                    processed_img = cv2.divide(gray, 255-blur, scale=256.0)
                
                elif effect == "Cartoon":
                    # Process edges
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.medianBlur(gray, 5)
                    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
                    # Process color
                    color = cv2.bilateralFilter(original_img, 9, 250, 250)
                    # Combine edges and color
                    processed_img = cv2.bitwise_and(color, color, mask=edges)
                
                elif effect == "HDR Effect":
                    processed_img = cv2.detailEnhance(original_img, sigma_s=12, sigma_r=0.15)

            # Display processed image
            if len(processed_img.shape) == 3:
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            else:
                st.image(processed_img, clamp=True)

            # Add download button for processed image
            if st.button("Download Processed Image"):
                if len(processed_img.shape) == 3:
                    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                else:
                    processed_img_rgb = processed_img
                # Convert to PIL image and prepare for download
                pil_img = Image.fromarray(processed_img_rgb)
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='PNG')
                st.download_button(
                    label="Download Image",
                    data=img_bytes.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()