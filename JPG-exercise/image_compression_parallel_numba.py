import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import njit, prange

# Precompute the DCT matrix once
@njit(cache=True)
def dct_orthonormal_matrix(N):
    C = np.zeros((N, N))
    for k in range(N):
        alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            C[k, n] = alpha * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return C

# JPEG quantization matrix
q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

# JIT-accelerated blockwise DCT, quantization, and IDCT
@njit(parallel=True, cache=True)
def process_blocks(image, dct_matrix, idct_matrix, q):
    h, w = image.shape
    output = np.zeros_like(image)
    block_size = 8
    h_blocks = h // block_size
    w_blocks = w // block_size

    for bi in prange(h_blocks):
        for bj in range(w_blocks):
            i = bi * block_size
            j = bj * block_size

            block = image[i:i+block_size, j:j+block_size] - 128.0

            # DCT
            temp1 = dct_matrix @ block
            dct_block = temp1 @ dct_matrix.T

            # Quantize
            quantized_block = np.round(dct_block / q)

            # Inverse quantization
            inv_block = quantized_block * q

            # IDCT
            temp2 = idct_matrix @ inv_block
            idct_block = temp2 @ idct_matrix.T + 128.0

            # Clip and store
            idct_block = np.clip(np.round(idct_block), 0, 255)
            output[i:i+block_size, j:j+block_size] = idct_block.astype(np.uint8)

    return output


# Load grayscale image
image_path = './JPG-exercise/image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = np.float64(image)

# Padding image to be multiple of 8x8 if needed
height, width = image.shape
pad_h = (8 - height % 8) % 8
pad_w = (8 - width % 8) % 8
image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

# Generate DCT and IDCT matrices
dct_matrix = dct_orthonormal_matrix(8)
idct_matrix = dct_matrix.T

# Process image
reconstructed = process_blocks(image_padded, dct_matrix, idct_matrix, q)

# Crop back to original size
reconstructed = reconstructed[:height, :width]

# Show original vs reconstructed
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap='gray')
plt.title("Reconstructed (Numba)")
plt.axis('off')

plt.show()
