import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

def global_contrast_enhancement(image, sigmoid_alpha):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    sigmoid_hist = 1 / (1 + np.exp(-(hist - np.mean(hist)) / np.std(hist)))
    distribution_function = sigmoid_hist / np.sum(sigmoid_hist)
    cdf = np.cumsum(distribution_function)
    lookup_table = np.uint8(255 * cdf)
    contrast_enhanced_image = cv2.LUT(gray_image, lookup_table)
    contrast_enhanced_image = cv2.cvtColor(contrast_enhanced_image, cv2.COLOR_GRAY2BGR)
    return contrast_enhanced_image

def adjust_dct_coefficients(image, dct_multiplier):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv_image[:,:,0].astype(np.float32) - 128.0
    dct_coefficients = cv2.dct(y_channel)
    dct_coefficients *= dct_multiplier
    y_channel_adjusted = cv2.idct(dct_coefficients) + 128.0
    yuv_image[:,:,0] = np.clip(y_channel_adjusted, 0, 255).astype(np.uint8)
    enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return enhanced_image

def on_button_click(b):
    global contrast_enhancement_alpha, dct_multiplier
    contrast_enhancement_alpha = alpha_slider.value
    dct_multiplier = dct_multiplier_slider.value

    # Check if an image is uploaded
    if uploaded_file.value:
        # Get the uploaded image
        uploaded_image = list(uploaded_file.value.values())[0]
        input_image = cv2.imdecode(np.frombuffer(uploaded_image['content'], dtype=np.uint8), -1)

        # Apply global contrast enhancement
        contrast_enhanced_image = global_contrast_enhancement(input_image.copy(), contrast_enhancement_alpha)

        # Apply empirical adjustments to DCT coefficients
        result_image = adjust_dct_coefficients(contrast_enhanced_image, dct_multiplier)

        # Display the original, contrast-enhanced, and result images using matplotlib
        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(cv2.cvtColor(contrast_enhanced_image, cv2.COLOR_BGR2RGB))
        plt.title('Contrast Enhanced Image')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Result Image')
        plt.axis('off')

        plt.show()
    else:
        print("Please upload an image.")

# Upload image
uploaded_file = widgets.FileUpload()
display(uploaded_file)

# Parameters
alpha_slider = widgets.FloatSlider(min=0.1, max=5, step=0.1, value=1.0, description='Alpha:')
dct_multiplier_slider = widgets.FloatSlider(min=0.1, max=5, step=0.1, value=1.0, description='DCT Multiplier:')

# Button
button = widgets.Button(description='Apply Enhancements')
button.on_click(on_button_click)

# Display widgets
display(alpha_slider, dct_multiplier_slider, button)
