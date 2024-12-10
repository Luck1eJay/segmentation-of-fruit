import os
from PIL import Image

def sample_image(image_path, save_folder, window_size=36):
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size

    # 创建保存文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 初始化计数器
    count = 0
    stride = window_size * 1  # 设置步长为窗口大小的3倍

    # 遍历图片并采样
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # 裁剪图像
            box = (x, y, x + window_size, y + window_size)
            cropped_img = img.crop(box)

            # 保存图片
            save_path = os.path.join(save_folder, f"{count}.png")
            cropped_img.save(save_path)

            # 增加计数器
            count += 1

    print(f"Saved {count} images.")

# 使用示例
image_path = "img1.jpg"  # 输入图片路径
save_folder = "Generatedataset"  # 保存采样图片的文件夹

sample_image(image_path, save_folder)