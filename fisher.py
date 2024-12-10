import os
import numpy as np
from PIL import Image

# 定义颜色
colors = {
    'background': [0, 0, 0],  # Black
    'grape': [255, 0, 0],     # Red
    'apple': [0, 255, 0],     # Green
    'melon': [0, 0, 255]      # Blue
}

# 加载样本
def load_samples(folder):
    samples = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(folder, filename))
            samples.append(np.array(img).reshape(-1, 3))
    return np.vstack(samples)

# 计算均值和协方差矩阵
def calculate_statistics(samples):
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    return mean, cov

# Fisher 判别函数
def fisher_discriminant(pixel, means, covs, priors):
    scores = []
    for mean, cov, prior in zip(means, covs, priors):
        inv_cov = np.linalg.inv(cov)
        score = np.dot(np.dot((pixel - mean).T, inv_cov), (pixel - mean)) + np.log(np.linalg.det(cov)) - 2 * np.log(prior)
        scores.append(score)
    return np.argmin(scores)

# 加载样本并计算统计量
categories = ['background', 'grape', 'apple', 'melon']
means = []
covs = []
priors = []
base_path = 'Generatedataset'
total_samples = 0

for category in categories:
    samples = load_samples(os.path.join(base_path, category))
    mean, cov = calculate_statistics(samples)
    means.append(mean)
    covs.append(cov)
    priors.append(len(samples))
    total_samples += len(samples)

priors = [prior / total_samples for prior in priors]

# 读取待分类图像
image_path = 'img3.jpg'
image = Image.open(image_path)
image_np = np.array(image)

# 对每个像素进行分类
classified_image = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

for i in range(image_np.shape[0]):
    for j in range(image_np.shape[1]):
        pixel = image_np[i, j]
        classified_image[i, j] = fisher_discriminant(pixel, means, covs, priors)

# 创建每个类别的掩码
colored_mask = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
for i, category in enumerate(categories):
    colored_mask[classified_image == i] = colors[category]

# 保存掩码图像
colored_mask_image = Image.fromarray(colored_mask)
colored_mask_image.save('mask_fisher3.png')

print("Classification and mask generation completed.")