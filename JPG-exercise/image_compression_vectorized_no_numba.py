import numpy as np
import cv2
import matplotlib.pyplot as plt

def dct_orthonormal_matrix(N):
    C = np.zeros((N, N))
    for k in range(N):
        alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            C[k, n] = alpha * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return C

def quantize(block, q):
    return np.int8(np.round(block / q))

def dequantize(block, q):
    return block * q

def safe_uint8(block):
    return np.uint8(np.clip(block, 0, 255))
# Block-by-block DCT and IDCT (sequential version)
def process_blocks_seq(image, dct_matrix, idct_matrix, q):
    h, w = image.shape
    output = np.zeros_like(image)
    block_size = 8

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size] - 128.0

            dct_block = dct_matrix @ block @ dct_matrix.T
            quantized_block = quantize(dct_block, q)

            inv_block = dequantize(quantized_block, q)
            idct_block = idct_matrix @ inv_block @ idct_matrix.T + 128.0

            output[i:i+block_size, j:j+block_size] = safe_uint8(idct_block)

    return output

# Load grayscale image
image_path = './JPG-exercise/image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = np.float64(image)

# Pad image
h, w = image.shape
pad_h = (8 - h % 8) % 8
pad_w = (8 - w % 8) % 8
image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

# DCT & IDCT matrices
dct_matrix = dct_orthonormal_matrix(8)
idct_matrix = dct_matrix.T

# Quantization table
q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float64)

# Process image (sequential)
reconstructed = process_blocks_seq(image_padded, dct_matrix, idct_matrix, q)
reconstructed = reconstructed[:h, :w]
image = np.uint8(image)

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap='gray')
plt.title("Reconstructed (Vectorized Ops)")
plt.axis('off')
plt.show()
