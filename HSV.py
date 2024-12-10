#coding=UTF-8
from PIL import Image, ImageDraw
import numpy as np


def rgb_to_hsv(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    mAx = max(r, g, b)
    mIn = min(r, g, b)
    h = s = v = mAx

    c = mAx - mIn
    if c == 0:
        h = 0
    elif mAx == r:
        h = (60 * ((g - b) / c) + 360) % 360
    elif mAx == g:
        h = (60 * ((b - r) / c) + 120) % 360
    elif mAx == b:
        h = (60 * ((r - g) / c) + 240) % 360

    s = 0 if mAx == 0 else (c / mAx)
    v = mAx

    return h, s, v

def create_masks(image_path):

    image = Image.open(image_path)
    image = image.convert('RGB')
    
    data = np.array(image)

    # Initialize masks
    red_mask = np.zeros(data.shape[:2], dtype=np.uint8)
    green_mask = np.zeros(data.shape[:2], dtype=np.uint8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r, g, b = data[i, j]
            h, s, v = rgb_to_hsv(r, g, b)

            # red fruit
            if ((h >= 0 and h <= 17) or (h >= 340 and h <= 360)) and s > 0.1 and v > 0.1:
                red_mask[i, j] = 255
            # green fruit
            elif (h >= 17 and h <= 180) and s > 0.1 and v > 0.1:
                green_mask[i, j] = 255

    #create masks
    red_fruits = np.zeros_like(data)
    green_fruits = np.zeros_like(data)

    red_fruits[red_mask == 255] = data[red_mask == 255]
    green_fruits[green_mask == 255] = data[green_mask == 255]

    red_fruits_image = Image.fromarray(red_fruits)
    green_fruits_image = Image.fromarray(green_fruits)

    return red_fruits_image, green_fruits_image


def calculate_histogram(channel_data):
    histogram = np.zeros(256, dtype=int)
    for value in channel_data.ravel():
        histogram[value] += 1
    histogram[255]=0
    return histogram


def draw_histogram(histogram, color):
    hist_height = 256
    hist_width = 256
    hist_image = Image.new("RGB", (hist_width, hist_height), "white")
    draw = ImageDraw.Draw(hist_image)

    max_value = max(histogram)
    scale = hist_height / max_value

    for i in range(256):
        value = histogram[i]
        draw.line([(i, hist_height), (i, hist_height - value * scale)], fill=color)

    return hist_image


def show_histograms_and_grayscale(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    data = np.array(image)

    # Separate the color channels
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    r_gray = Image.fromarray(r)
    g_gray = Image.fromarray(g)
    b_gray = Image.fromarray(b)

    # Calculate histograms
    r_hist = calculate_histogram(r)
    g_hist = calculate_histogram(g)
    b_hist = calculate_histogram(b)

    # Draw histograms
    r_hist_image = draw_histogram(r_hist, 'red')
    g_hist_image = draw_histogram(g_hist, 'green')
    b_hist_image = draw_histogram(b_hist, 'blue')



image_path = 'm.jpg'
red_fruits, green_fruits = create_masks(image_path)
show_histograms_and_grayscale(image_path)
# visualize
red_fruits.show(title='apple')
#green_fruits.show(title='Cucumis melo')

