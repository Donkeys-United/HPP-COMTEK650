import numpy as np
import cv2
import matplotlib.pyplot as plt

def dct_1d(vector):
    N = len(vector)
    result = np.zeros(N)
    for k in range(N):
        sum_val = 0.0
        alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            sum_val += alpha * vector[n] * np.cos(((2*n + 1) * k * np.pi) / (2 * N))
        result[k] = sum_val
    return result

def dct_2d_block(block):
    M, N = block.shape
    temp = np.zeros((M, N))

    # DCT on rows
    for i in range(M):
        temp[i, :] = dct_1d(block[i, :])

    # DCT on columns
    result = np.zeros((M, N))
    for j in range(N):
        result[:, j] = dct_1d(temp[:, j])

    return result

def idct_1d(vector):
    N = len(vector)
    result = np.zeros(N)
    for n in range(N):
        sum_val = 0.0
        for k in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            sum_val += alpha * vector[k] * np.cos(((2*n + 1) * k * np.pi) / (2 * N))
        result[n] = sum_val
    return result

def idct_2d_block(block):
    M, N = block.shape
    temp = np.zeros((M, N))

    # IDCT on rows
    for i in range(M):
        temp[i, :] = idct_1d(block[i, :])

    result = np.zeros((M, N))
    for j in range(N):
        result[:, j] = idct_1d(temp[:, j])

    return result

# Load the image in grayscale using OpenCV
image_path = './JPG-exercise/image.jpg'  # Provide the image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Normalize the image (to the range [0, 255])
image = np.float64(image)

# Slice the image into 8x8 blocks
height, width = image.shape
blocks = []

# Extracting 8x8 blocks
for i in range(0, height, 8):
    for j in range(0, width, 8):
        block = image[i:i+8, j:j+8]
        blocks.append(block)

q = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]])
transformed_blocks = []
quantized_blocks = []

for block in blocks:
    shifted_block = block - 128  # Centering around zero (same as you did earlier)
    transformed_block = dct_2d_block(shifted_block)  # Apply DCT
    quantized_block = np.round(transformed_block / q).astype(np.int8)  # Quantize the coefficients
    transformed_blocks.append(quantized_block)
    quantized_blocks.append(quantized_block)

# Reconstruct the image from the quantized blocks
reconstructed_image = np.zeros_like(image)

for i, block in enumerate(quantized_blocks):
    # Inverse quantization
    inverse_quantized_block = block * q  # Multiply by the quantization matrix
    # Apply IDCT
    reconstructed_block = idct_2d_block(inverse_quantized_block) + 128  # Add back the DC offset
    reconstructed_block = np.clip(reconstructed_block, 0, 255)
    # Place the reconstructed block back into the image
    row = ((i * 8) // width)*8
    col = (i * 8) % width
    reconstructed_image[row:row+8, col:col+8] = np.round(reconstructed_block).astype(np.uint8)

# Compare original and reconstructed images
# Display both images side by side using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title("Reconstructed Image (Compressed)")
plt.axis('off')

plt.show()

# You can also save the reconstructed image using OpenCV
#cv2.imwrite('reconstructed_image.jpg', reconstructed_image)