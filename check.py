import numpy as np
from PIL import Image

# 定义颜色
colors = {
    'background': [0, 0, 0],  # Black
    'grape': [255, 0, 0],     # Red
    'apple': [0, 255, 0],     # Green
    'melon': [0, 0, 255]      # Blue
}

# 加载图像
label_image_path = 'LABELimg2.png'
mask_image_path = 'mask_fisher2.png'
label_image = Image.open(label_image_path).convert('RGB')
mask_image = Image.open(mask_image_path).convert('RGB')

label_np = np.array(label_image)
mask_np = np.array(mask_image)

# 初始化计数器和可视化结果
mismatch_counts = {
    'grape': 0,
    'apple': 0,
    'melon': 0
}
color_totals = {
    'grape': 0,
    'apple': 0,
    'melon': 0
}
intersection_counts = {
    'grape': 0,
    'apple': 0,
    'melon': 0
}
union_counts = {
    'grape': 0,
    'apple': 0,
    'melon': 0
}
mismatch_visual = {
    'grape': np.zeros_like(label_np),
    'apple': np.zeros_like(label_np),
    'melon': np.zeros_like(label_np)
}

# 遍历每个像素
for i in range(label_np.shape[0]):
    for j in range(label_np.shape[1]):
        label_pixel = label_np[i, j]
        mask_pixel = mask_np[i, j]

        if np.array_equal(label_pixel, colors['grape']):
            color_totals['grape'] += 1
            if np.array_equal(mask_pixel, colors['grape']):
                intersection_counts['grape'] += 1
            else:
                mismatch_counts['grape'] += 1
                mismatch_visual['grape'][i, j] = [255, 255, 255]  # Mark mismatch as white
            union_counts['grape'] += 1
        elif np.array_equal(label_pixel, colors['apple']):
            color_totals['apple'] += 1
            if np.array_equal(mask_pixel, colors['apple']):
                intersection_counts['apple'] += 1
            else:
                mismatch_counts['apple'] += 1
                mismatch_visual['apple'][i, j] = [255, 255, 255]  # Mark mismatch as white
            union_counts['apple'] += 1
        elif np.array_equal(label_pixel, colors['melon']):
            color_totals['melon'] += 1
            if np.array_equal(mask_pixel, colors['melon']):
                intersection_counts['melon'] += 1
            else:
                mismatch_counts['melon'] += 1
                mismatch_visual['melon'][i, j] = [255, 255, 255]  # Mark mismatch as white
            union_counts['melon'] += 1

# 计算并集
for i in range(mask_np.shape[0]):
    for j in range(mask_np.shape[1]):
        mask_pixel = mask_np[i, j]
        if np.array_equal(mask_pixel, colors['grape']) and not np.array_equal(label_np[i, j], colors['grape']):
            union_counts['grape'] += 1
        elif np.array_equal(mask_pixel, colors['apple']) and not np.array_equal(label_np[i, j], colors['apple']):
            union_counts['apple'] += 1
        elif np.array_equal(mask_pixel, colors['melon']) and not np.array_equal(label_np[i, j], colors['melon']):
            union_counts['melon'] += 1

# 计算比例
mismatch_ratios = {category: mismatch_counts[category] / color_totals[category] if color_totals[category] > 0 else 0 for category in mismatch_counts}

# 计算 IoU
iou_scores = {category: intersection_counts[category] / union_counts[category] if union_counts[category] > 0 else 0 for category in intersection_counts}

# 保存可视化结果
for category in mismatch_visual:
   mismatch_image = Image.fromarray(mismatch_visual[category])
   mismatch_image.save(f'mismatch_{category}_fisher.png')

# 输出结果
print("Mismatch counts:")
print(f"Grape (Red): {mismatch_counts['grape']} / {color_totals['grape']} = {mismatch_ratios['grape']:.2%}")
print(f"Apple (Green): {mismatch_counts['apple']} / {color_totals['apple']} = {mismatch_ratios['apple']:.2%}")
print(f"Melon (Blue): {mismatch_counts['melon']} / {color_totals['melon']} = {mismatch_ratios['melon']:.2%}")

print("IoU scores:")
print(f"Grape (Red): {iou_scores['grape']:.2%}")
print(f"Apple (Green): {iou_scores['apple']:.2%}")
print(f"Melon (Blue): {iou_scores['melon']:.2%}")