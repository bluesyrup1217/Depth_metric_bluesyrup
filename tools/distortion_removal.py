import cv2
import numpy as np
import json
import os

def remove_distortion(input_path, output_path, camera_matrix, dist_coeffs, image_size, scale_factor=1.0, use_fisheye=False):
    """
    广角相机图像去畸变函数
    
    参数:
    input_path: 输入图像路径
    output_path: 输出图像路径
    camera_matrix: 相机内参矩阵 (3x3)
    dist_coeffs: 畸变系数
    image_size: 输入图像尺寸 (宽, 高)
    scale_factor: 缩放因子，用于调整去畸变后的图像大小
    use_fisheye: 是否使用鱼眼相机模型 (适用于大视场角相机)
    
    返回:
    bool: 去畸变是否成功
    """
    # 读取输入图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误: 无法读取图像 {input_path}")
        return False
    
    # 创建新的相机内参矩阵
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, scale_factor
    )
    
    width, height = image_size
    mapx, mapy = None, None
    
    if use_fisheye:
        # 使用鱼眼相机模型去畸变
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, 
            (width, height), cv2.CV_32FC1
        )
    else:
        # 使用针孔相机模型去畸变
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, 
            (width, height), cv2.CV_32FC1
        )
    
    # 应用去畸变映射
    undistorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    # 裁剪到有效区域
    x, y, w, h = roi
    valid_undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    # 将有效区域图像填充回原始分辨率（四周用黑边填充）
    # 创建一个与原图像相同大小的黑色图像
    final_img = np.zeros_like(undistorted_img)
    
    # 计算有效图像在黑色图像中的位置（居中放置）
    start_x = (final_img.shape[1] - valid_undistorted_img.shape[1]) // 2
    start_y = (final_img.shape[0] - valid_undistorted_img.shape[0]) // 2
    
    # 将有效图像放置到黑色图像的中心
    final_img[start_y:start_y+valid_undistorted_img.shape[0], start_x:start_x+valid_undistorted_img.shape[1]] = valid_undistorted_img
    
    # 更新为最终图像
    undistorted_img = final_img
    
    # 保存去畸变后的图像
    cv2.imwrite(output_path, undistorted_img)
    print(f"去畸变后的图像已保存到 {output_path}")
    print(f"去畸变后图像尺寸: {undistorted_img.shape[1]}x{undistorted_img.shape[0]}")
    print(f"有效区域尺寸: {valid_undistorted_img.shape[1]}x{valid_undistorted_img.shape[0]}")
    print(f"填充后尺寸: {final_img.shape[1]}x{final_img.shape[0]}")
    
    return True

# ============================================
# 主程序 - 使用变量方式配置参数
# ============================================

# ---------- 输入输出配置 ----------
# 输入图像路径
input_image_path = "/home/lixin/code/MoGe/data/4mm/x9e3mHPUDpzbEHHH4_171908_9401790713934496.jpg"

# 自动生成输出图像路径：原图像名称 + _distortion_removal
# 获取原始文件名和扩展名
original_filename = os.path.basename(input_image_path)
original_name, original_ext = os.path.splitext(original_filename)

# 输出目录
output_dir = "/home/lixin/code/MoGe/output"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 生成输出图像路径
output_filename = f"{original_name}_distortion_removal{original_ext}"
output_image_path = os.path.join(output_dir, output_filename)

# ---------- 相机参数配置 ----------
# 是否使用鱼眼相机模型 (适用于大视场角相机)
use_fisheye_model = False

# 缩放因子，用于调整去畸变后的图像大小
# 1.0: 保持原始尺寸
# >1.0: 放大图像，保留更多边缘信息
# <1.0: 缩小图像，可能会裁剪更多边缘
scale_factor = 1.0

# 原始1080p相机参数
original_width = 1920
original_height = 1080

# 当前使用的图像尺寸（已缩放到1280x720）
current_width = 1280
current_height = 720

# 计算缩放因子
scale_factor_internal = current_width / original_width  # 1280/1920 = 2/3

# 原始1080p内参
original_fx = 1223.2489013671875
original_fy = 1217.8367919921875
original_cx = 971.3410034179688
original_cy = 533.7720336914062

# 计算缩放后的内参（适用于1280x720图像）
fx = original_fx * scale_factor_internal
fy = original_fy * scale_factor_internal
cx = original_cx * scale_factor_internal
cy = original_cy * scale_factor_internal

# 相机内参矩阵 (3x3)
camera_matrix = np.array([
    [fx, 0, cx],  # fx, 0, cx
    [0, fy, cy],  # 0, fy, cy
    [0, 0, 1]     # 0, 0, 1
])

# 畸变系数
# 如果use_fisheye_model=True，使用鱼眼模型的畸变系数 [k1, k2, k3, k4, k5, k6]
# 如果use_fisheye_model=False，使用针孔模型的畸变系数 [k1, k2, p1, p2, k3]
if use_fisheye_model:
    # 鱼眼相机模型畸变系数
    dist_coeffs = np.array([-0.2955333888530731, 0.10201843827962875, 0, 0, 0, 0])
else:
    # 针孔相机模型畸变系数 [k1, k2, p1, p2, k3]
    dist_coeffs = np.array([-0.2955333888530731, 0.10201843827962875, 0.000291781616397202, -0.0004300259461160749, 0])

# ---------- 执行去畸变 ----------
# 读取输入图像以获取尺寸
img = cv2.imread(input_image_path)
if img is None:
    print(f"错误: 无法读取图像 {input_image_path}")
    exit(1)

height, width = img.shape[:2]
image_size = (width, height)

# 打印配置信息
print("去畸变配置信息:")
print("====================")
print(f"输入图像: {input_image_path}")
print(f"输出图像: {output_image_path}")
print(f"图像尺寸: {width}x{height}")
print(f"使用模型: {'鱼眼' if use_fisheye_model else '针孔'}")
print(f"缩放因子: {scale_factor}")
print("\n相机内参矩阵:")
print(camera_matrix)
print("\n畸变系数:")
print(dist_coeffs)
print("====================")

# 执行去畸变
success = remove_distortion(
    input_image_path,
    output_image_path,
    camera_matrix,
    dist_coeffs,
    image_size,
    scale_factor,
    use_fisheye_model
)

# 程序结束
if success:
    print("\n去畸变完成！")
else:
    print("\n去畸变失败！")
    exit(1)
