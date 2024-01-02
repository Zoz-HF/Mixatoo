import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_magnitude_and_phase(image):
    # Compute the Fourier transform of the image
    f_transform = np.fft.fft2(image)

    # Get the magnitude and phase of the Fourier transform
    magnitude = np.abs(f_transform)
    phase = np.angle(f_transform)

    return magnitude, phase


def combine_images(images, selected_magnitudes, selected_phases, weights):
    combined_magnitude = np.zeros_like(images[0], dtype=np.float64)
    combined_phase = np.zeros_like(images[0], dtype=np.float64)

    # Normalize weights so that they sum to 1
    weights = np.array(weights) 

    for i in range(len(images)):
        magnitude, phase = get_magnitude_and_phase(images[i])

        if selected_magnitudes[i]:
            combined_magnitude += weights[i] * magnitude
        else:
            combined_magnitude += weights[i] * np.ones_like(magnitude)

        if selected_phases[i]:
            combined_phase += weights[i] * phase
        else:
            combined_phase += weights[i] * np.zeros_like(phase)

    # Combine magnitude and phase
    combined_f_transform = combined_magnitude * np.exp(1j * combined_phase)

    # Inverse Fourier Transform
    combined_image = np.fft.ifft2(combined_f_transform).real

    # Convert the result back to uint8
    # combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    return combined_image


# Load your four images
image1 = cv2.imread('image 1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image 2.jpg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('image 3.jpg', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('image 4.jpg', cv2.IMREAD_GRAYSCALE)

target_size = (500, 500)
image1 = cv2.resize(image1, target_size)
image2 = cv2.resize(image2, target_size)
image3 = cv2.resize(image3, target_size)
image4 = cv2.resize(image4, target_size)

# Choose the weights for each image
weights = [1.0, 1.0, 0.0, 0.0]

# Choose whether to use magnitude or phase for each image

selected_magnitudes = [False, True, False, False]
selected_phases = [False, True, False, False]

# Combine images
combined_image = combine_images([image1, image2, image3, image4], selected_magnitudes, selected_phases, weights)

# Display the result
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Image')
plt.show()
