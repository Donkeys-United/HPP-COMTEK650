import cv2
import numpy as np
from scipy.fftpack import dct, idct

# Apply 2D DCT to a block
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def get_standard_quantization_matrix():
    return np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ], dtype=np.float32)



img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)
img_centered = img - 128
height, width = img_centered.shape

Q = get_standard_quantization_matrix()
quantized_blocks = []


for row in range(0,height,8):
    for col in range(0, width, 8):
        block = img_centered[row:row+8, col:col+8]
        block_dct = dct2(block)
        YQ = np.round(block_dct/Q)
        quantized_blocks.append(YQ)



# Reconstruct image from quantized blocks
reconstructed = np.zeros_like(img_centered)

index = 0
for row in range(0, height, 8):
    for col in range(0, width, 8):
        YQ = quantized_blocks[index]
        Y = YQ * Q  # Dequantize
        block_reconstructed = idct2(Y)
        reconstructed[row:row+8, col:col+8] = block_reconstructed
        index += 1

# Add 128 back to return to original pixel range
reconstructed += 128

# Clip values to valid image range
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

# Show image
cv2.imshow("Reconstructed Image", reconstructed)
cv2.waitKey(0)
cv2.destroyAllWindows()

combined = np.hstack((img.astype(np.uint8), reconstructed))
cv2.imshow("Original (left) vs Compressed (right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()