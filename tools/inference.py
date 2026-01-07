import sys
sys.path.append("/home/lixin/code/MoGe/")
import cv2
import torch
import numpy as np
import os
import math
from moge.model.v2 import MoGeModel # Let's try MoGe-2


device = torch.device("cuda:3")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image_path = "/home/lixin/code/MoGe/output/x9e3mHPUDpzbEHHH4_171908_9401790713934496_distortion_removal.jpg"
img = cv2.imread(input_image_path)
input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                        

# 输入图像尺寸
img_width, img_height = img.shape[1], img.shape[0]
print(f"输入图像尺寸: {img_width}x{img_height}")

# 原始相机内参 (1920x1080)
original_params = {
    "fx": 1223.2489013671875,
    "fy": 1217.8367919921875,
    "cx": 971.3410034179688,
    "cy": 533.7720336914062,
    "cawidth": 1920,
    "caheight": 1080,
}

# 计算缩放因子
scale_w = img_width / original_params["cawidth"]
scale_h = img_height / original_params["caheight"]
scaled_fx = original_params["fx"] * scale_w
scaled_fy = original_params["fy"] * scale_h
scaled_cx = original_params["cx"] * scale_w
scaled_cy = original_params["cy"] * scale_h
fov_x_rad = 2 * math.atan(img_width / (2 * scaled_fx))
fov_x_deg = math.degrees(fov_x_rad)
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
whether_using_fov = False
#---------------------------------------------------
if whether_using_fov:
    print(f"使用FOV冻结内参: {fov_x_deg:.2f}")
    output = model.infer(input_image, fov_x=fov_x_deg)
else:
    print("不使用FOV")
    output = model.infer(input_image)
#---------------------------------------------------
depth_map = output["depth"].cpu().numpy()
finite_mask = np.isfinite(depth_map)
if np.any(finite_mask):
    max_valid_depth = depth_map[finite_mask].max()
    depth_map[~finite_mask] = max_valid_depth

# 获取深度图的分辨率
height, width = depth_map.shape
print(f"深度图分辨率: {width}x{height}")

# 通过坐标点获取深度值的功能
# 示例坐标点 (x, y) - 可以根据需要修改或改为用户输入
# 注意：OpenCV的坐标系统是 (x, y) = (列, 行)
x_coord = 1136 # 中心点
y_coord = 498  # 中心点

# 验证坐标点是否在有效范围内
if 0 <= x_coord < width and 0 <= y_coord < height:
    depth_value = depth_map[y_coord, x_coord]
    print(f"坐标点 ({x_coord}, {y_coord}) 的深度值: {depth_value:.2f}")
else:
    print(f"错误：坐标点 ({x_coord}, {y_coord}) 超出深度图范围")

# 将深度图归一化到[0, 255]范围
normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# 生成伪彩色图
colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
# 提取原图文件名并构建新文件名
filename = os.path.basename(input_image_path)
name_without_ext, ext = os.path.splitext(filename)
output_filename = f"{name_without_ext}_heat{ext}"
output_path = os.path.join("output", output_filename)
cv2.imwrite(output_path, colored_depth)