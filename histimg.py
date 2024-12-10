import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取图像
image_path = 'img3.jpg'
image = Image.open(image_path)
image_np = np.array(image)

# 分离 RGB 通道
r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]

# 手动计算直方图
def calculate_histogram(channel):
    hist = np.zeros(256, dtype=int)
    for value in channel.flatten():
        hist[value] += 1
    return hist

r_hist = calculate_histogram(r)
g_hist = calculate_histogram(g)
b_hist = calculate_histogram(b)

# 绘制直方图
plt.figure(figsize=(10, 5))
plt.plot(range(256), r_hist, color='red', alpha=0.6, label='Red')
plt.plot(range(256), g_hist, color='green', alpha=0.6, label='Green')
plt.plot(range(256), b_hist, color='blue', alpha=0.6, label='Blue')

# 添加图例和标签
plt.legend(loc='upper right')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('RGB Histogram')

# 显示图像
plt.show()