"""
Discrete Cosine Transform
===========================

This example shows how to use the :py:class:`pylops.signalprocessing.DCT` operator.
This operator performs Discrete Cosine Transform on the input array (single or multi-dimension)

"""

import matplotlib.pyplot as plt
import numpy as np

from pylops.signalprocessing.dct import DCT

plt.close("all")


###############################################################################
# Let's define a !D array x of increasing numbers

n = 21
x = np.arange(n) + 1


###############################################################################
# Next we create the DCT operator with the shape of our input array as parameter
# Store the transformed array in `y`
# To perform inverse transform, use `Dct.H`

Dct = DCT(dims=x.shape)
y = Dct * x

plt.figure(figsize=(8, 5))
plt.plot(x, label="input array")
plt.plot(y, label="transformed array")
plt.title("1D Discrete Cosine Transform")
plt.tight_layout()
plt.legend()

################################################################################
# Applying DCT to a sine wave

cycles = 2
resolution = 100

length = np.pi * 2 * cycles
s = np.sin(np.arange(0, length, length / resolution))
Dct = DCT(dims=s.shape)
y = Dct * s

plt.figure(figsize=(8, 5))
plt.plot(s, label="sine wave")
plt.plot(y, label="dct of sine wave")
plt.title("Discrete Cosine Transform of Sine wave")
plt.tight_layout()
plt.legend()


###############################################################################
# Discrete Cosine Transform is used in lossy image compression due to it's strong energy compaction
# nature. Here is an example of DCT being used for image compression.
# Note: This code is just an example and may not provide the best results for all images.
# You may need to adjust the threshold value to get better results.

img = np.load("../testdata/python.npy")[::5, ::5, 0]
Dct = DCT(dims=img.shape)
dct_img = Dct * img

# Set a threshold for the DCT coefficients to zero out
threshold = np.percentile(np.abs(dct_img), 70)
dct_img[np.abs(dct_img) < threshold] = 0

# Inverse DCT to get back the image
compressed_img = Dct.H * dct_img

# Plot original and compressed images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original Image")
ax[1].imshow(compressed_img, cmap="gray")
ax[1].set_title("Compressed Image")
