import os
import numpy as np
from PIL import Image

def load_samples(folder):
    samples = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(folder, filename))
            samples.append(np.array(img).reshape(-1, 3))
    return np.vstack(samples)

def calculate_statistics(samples):
    mean = np.mean(samples, axis=0)
    var = np.var(samples, axis=0)
    return mean, var

def naive_bayes_classifier(pixel, means, vars):
    probabilities = []
    for mean, var in zip(means, vars):
        prob = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((pixel - mean) ** 2) / var)
        probabilities.append(prob)
    return np.argmax(probabilities)

# Load samples and calculate statistics
categories = ['background', 'grape', 'apple', 'melon']
means = []
vars = []
base_path = 'Generatedataset'
for category in categories:
    samples = load_samples(os.path.join(base_path, category))
    mean, var = calculate_statistics(samples)
    means.append(mean)
    vars.append(var)

# Read the image to be classified
image_path = 'img1.jpg'
image = Image.open(image_path)
#image = Image.open(image_path).convert('RGB')

image_np = np.array(image)

# Classify each pixel(patch)
#block_size = 9
classified_image = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

for i in range(image_np.shape[0]):
    for j in range(image_np.shape[1]):
        pixel = image_np[i, j]
        classified_image[i, j] = naive_bayes_classifier(pixel, means, vars)
# for i in range(0, image_np.shape[0] - block_size + 1, block_size):
#     for j in range(0, image_np.shape[1] - block_size + 1, block_size):
#         block = image_np[i:i+block_size, j:j+block_size].reshape(-1, 3)
#         block_mean = np.mean(block, axis=0)
#         classified_image[i:i+block_size, j:j+block_size] = naive_bayes_classifier(block_mean, means, vars)

# Create masks for each category
masks = {category: np.zeros_like(classified_image, dtype=np.uint8) for category in categories}

colors = {
    'background':[0,0,0], # Black
    'grape':[255,0,0],    # Red
    'apple':[0,255,0],    # Green
    'melon':[0,0,255]     # Blue
}
colored_mask = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
for i, category in enumerate(categories):
    colored_mask[classified_image == i] = colors[category]
# for i in range(image_np.shape[0]):
#     for j in range(image_np.shape[1]):
#         if np.array_equal(image_np[i, j], [255, 255, 255]):
#             colored_mask[i, j] = [0, 0, 0]  # Set to black mask
#         else:
#             category_index = classified_image[i, j]
#             colored_mask[i, j] = colors[categories[category_index]]
# Save the masks
colored_mask_image = Image.fromarray(colored_mask)
colored_mask_image.save('maskm.png')

print("Classification and mask generation completed.")