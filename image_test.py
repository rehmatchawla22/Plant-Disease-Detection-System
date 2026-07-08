from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

folder = "dataset/Tomato___healthy"

image_name = os.listdir(folder)[0]

image_path = os.path.join(folder, image_name)

img = Image.open(image_path)

print("Original Size :", img.size)

# Resize
img = img.resize((224,224))

print("Resized Size :", img.size)

img_array = np.array(img)

# Normalize pixel values
img_array = img_array / 255.0

print("Min Value :", img_array.min())

print("Max Value :", img_array.max())

print("\nFirst Pixel After Normalization:")

print(img_array[0][0])

print("Shape :", img_array.shape)

plt.imshow(img)

plt.axis("off")

plt.show()