---
title: ImageLab Pro
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Advanced Image Processing Laboratory

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8.svg)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.3-FFA500.svg)](https://matplotlib.org/)

## 📋 Description

This application is a comprehensive image processing laboratory built with OpenCV, Streamlit, and other Python libraries. It provides a wide range of image manipulation and analysis capabilities through an intuitive web interface, allowing users to upload images and apply various transformations and effects in real-time.

## ✨ Features

- **Basic Operations:** Resize, rotate, flip, and adjust brightness/contrast of images
- **Filtering:** Apply various filters including blur, Gaussian, median, and bilateral filtering
- **Color Spaces:** Convert images between RGB, HSV, LAB, and YCrCb color spaces
- **Thresholding:** Apply different thresholding techniques including binary, adaptive, and Otsu's method
- **Morphological Operations:** Perform erosion, dilation, opening, closing, and other morphological transformations
- **Edge Detection:** Detect edges using Canny, Sobel, Laplacian, and Scharr operators
- **Feature Detection:** Identify key features using Harris Corner, Shi-Tomasi, and FAST algorithms
- **Histogram Operations:** Analyze and equalize image histograms to enhance contrast
- **Advanced Effects:** Apply artistic transformations like pencil sketch, cartoon effect, and HDR

## 🛠️ Technologies

- **Streamlit:** Interactive web interface with real-time updates
- **OpenCV:** Computer vision algorithms for image processing
- **NumPy:** Efficient numerical operations on image data
- **Matplotlib:** Visualization of image histograms and data
- **Pillow:** Image file handling and format conversion

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/StreamliteProject.git
cd StreamliteProject

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run the application
streamlit run app.py
```

### Application Workflow

1. **Upload an Image:** Use the file uploader in the sidebar to select an image
2. **Select Processing Category:** Choose from the available image processing categories
3. **Configure Parameters:** Adjust the parameters specific to the selected operation
4. **View Results:** See the processed image update in real-time
5. **Download:** Save the processed image if desired

## 📸 Image Processing Capabilities

### Basic Operations
- Image resizing with variable scaling
- Rotation with custom angle selection
- Horizontal and vertical flipping
- Brightness and contrast adjustment
- Color quantization for reducing color palettes

### Filtering and Enhancement
- Simple blur (averaging) filter
- Gaussian blur with adjustable kernel size
- Median filtering for noise reduction
- Bilateral filtering with edge preservation
- Custom kernel filtering (sharpen, edge detection, emboss)

### Advanced Analysis
- Multiple edge detection algorithms
- Feature point detection and highlighting
- Histogram visualization and manipulation
- Adaptive contrast enhancement

### Artistic Effects
- Pencil sketch conversion
- Cartoon-style rendering
- Detail enhancement (HDR-like effect)

## 🔧 Configuration

The application offers an intuitive UI for configuring parameters:
- Kernel sizes for various filters and operations
- Threshold values for edge detection and thresholding
- Intensity levels for effects and transformations
- Display options for visualization

## 💻 Project Structure

- `app.py`: Main application code with all image processing functions
- `requirements.txt`: Dependencies required to run the application

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenCV team for their comprehensive computer vision library
- Streamlit team for their easy-to-use web application framework
- All contributors to the Python data science and image processing ecosystem 