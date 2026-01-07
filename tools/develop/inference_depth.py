####################################
####
####Code editing: Xin Li (Steve)
####Edit time: 2026-01-06
####
####################################
import sys

sys.path.append("./")
import cv2
import torch
import numpy as np
import os
import math
import time
from moge.model.v2 import MoGeModel

class MoGe2_Depth:
    def __init__(self, moge2_model: str, image_size=(720, 1080), device="cuda:0"):
        self.model_name = moge2_model
        self.device = device
        self.img_size = image_size
        self.scale = 255  # 图片预处理的归一化数值
        self.inference_id = None
        self.save_dir_name = None
        try:
            print(f"Load model: {self.model_name}")
            self.model = MoGeModel.from_pretrained(self.model_name).to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            exit(1)

    def pre_precess(self, img):
        input_img = torch.tensor(
            img / self.scale, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)
        return input_img

    def gen_intrinsics(self, orin_intrinsics: dict):
        """
        生成图像对应的内参和fov_x

        :param self:
        :param orin_intrinsics: 相机内参，主要包含fx, fy, cx, cy和对应的推理图像shape
        :type orin_intrinsics: dict
        :orin_intrinsics example:
            original_params = {
                "fx": 1223.2489013671875,
                "fy": 1217.8367919921875,
                "cx": 971.3410034179688,
                "cy": 533.7720336914062,
                "cawidth": 1920,
                "caheight": 1080,
            }
        return fov_x, real_intrinsics
        """
        real_img_size = self.img_size
        H, W = int(real_img_size[0]), int(real_img_size[1])
        scale_w = W / orin_intrinsics["cawidth"]
        scale_h = H / orin_intrinsics["caheight"]
        scaled_fx = orin_intrinsics["fx"] * scale_w
        scaled_fy = orin_intrinsics["fy"] * scale_h
        scaled_cx = orin_intrinsics["cx"] * scale_w
        scaled_cy = orin_intrinsics["cy"] * scale_h
        real_intrinsics = {
            "fx": scaled_fx,
            "fy": scaled_fy,
            "cx": scaled_cx,
            "cy": scaled_cy,
            "cawidth": W,
            "caheight": H,
        }
        fov_x_rad = 2 * math.atan(W / (2 * scaled_fx))
        fov_x_deg = math.degrees(fov_x_rad)
        return fov_x_deg, real_intrinsics

    def inference_depth(
        self,
        img,
        input_intrinsics: None,
        dist_coeffs: None,
        frozen_intri: bool,
        save_visual_result=True,
    ):
        self.inference_id = time.strftime("%Y%m%d_%H:%M:%S")
        self.save_dir_name = f"output/{self.inference_id}"
        os.makedirs(self.save_dir_name, exist_ok=True)
        preprocess_time_begin = time.time()
        input_fov_x, real_intrinsics = self.gen_intrinsics(input_intrinsics)
        img_removal_dist = self.distortion_removal(
            img_in=img, real_intrinsics=real_intrinsics, dist_coeffs=dist_coeffs
        )
        input_img = self.pre_precess(img_removal_dist)
        preprocess_time_end = time.time()
        with torch.no_grad():
            infer_time_begin = time.time()
            if frozen_intri and input_intrinsics != None:
                print("******冻结内参推理******")
                img_depth = self.model.infer(input_img, fov_x=input_fov_x)
            else:
                print("******不冻结内参推理******")
                img_depth = self.model.infer(input_img)
            infer_time_end = time.time()
        postprocess_time_begin = time.time()
        depth_map = img_depth["depth"].cpu().numpy()
        finite_mask = np.isfinite(depth_map)
        if np.any(finite_mask):
            max_valid_depth = depth_map[finite_mask].max()
            depth_map[~finite_mask] = max_valid_depth
        points = img_depth["points"].cpu().numpy()
        postprocess_time_end = time.time()
        height_depthmap, width_depthmap = depth_map.shape
        # print(f"深度特征图分辨率: {width_depthmap}x{height_depthmap}")
        print(f"预处理时间: {preprocess_time_end - preprocess_time_begin:.2f}s")
        print(f"推理时间: {infer_time_end - infer_time_begin:.2f}s")
        print(f"后处理时间: {postprocess_time_end - postprocess_time_begin:.2f}s")
        if save_visual_result:
            saveresult_time_begin = time.time()
            self.save_depth_np(depth_map=depth_map)  # 保存深度图为numpy数组
            self.save_depth_map(depth_map=depth_map)  # 保存深度图
            self.save_cloudply_map(
                points=points, color_image=img_removal_dist
            )  # 保存点云为PLY格式
            saveresult_time_end = time.time()
            print(f"保存推理结果时间: {saveresult_time_end - saveresult_time_begin:.2f}s")
        else:
            print("Non save result image!")
        
        return depth_map

    def distortion_removal(
        self,
        img_in,
        real_intrinsics,
        dist_coeffs={
            "k0": -0.2955333888530731,
            "k1": 0.10201843827962875,
            "p1": 0.000291781616397202,
            "p2": -0.0004300259461160749,
            "k2": 0,
        },
        scale_factor=1.0,
    ):
        # 创建numpy矩阵
        camera_intrinsics = np.array(
            [
                [real_intrinsics["fx"], 0, real_intrinsics["cx"]],
                [0, real_intrinsics["fy"], real_intrinsics["cy"]],
                [0, 0, 1],
            ]
        )
        # 广角摄像头畸变系数（套用针孔模型
        camera_dist_coeffs = np.array(
            [
                dist_coeffs["k0"],
                dist_coeffs["k1"],
                dist_coeffs["p1"],
                dist_coeffs["p2"],
                dist_coeffs["k2"],
            ]
        )
        img = cv2.imread(img_in)
        height, width = img.shape[:2]
        image_size = (width, height)
        if img is None:
            print(f"错误: 无法读取图像 {input_path}")
            return False
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_intrinsics, camera_dist_coeffs, image_size, scale_factor
        )
        mapx, mapy = None, None
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_intrinsics,
            camera_dist_coeffs,
            np.eye(3),
            new_camera_matrix,
            image_size,
            cv2.CV_32FC1,
        )
        undistorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # 裁剪到有效区域
        x, y, w, h = roi
        valid_undistorted_img = undistorted_img[y : y + h, x : x + w]
        # 将有效区域图像填充回原始分辨率（四周用黑边填充）
        # 创建一个与原图像相同大小的黑色图像
        final_img = np.zeros_like(undistorted_img)
        # 计算有效图像在黑色图像中的位置（居中放置）
        start_x = (final_img.shape[1] - valid_undistorted_img.shape[1]) // 2
        start_y = (final_img.shape[0] - valid_undistorted_img.shape[0]) // 2
        # 将有效图像放置到黑色图像的中心
        final_img[
            start_y : start_y + valid_undistorted_img.shape[0],
            start_x : start_x + valid_undistorted_img.shape[1],
        ] = valid_undistorted_img
        disted_img_path = os.path.join(self.save_dir_name, f"distortioned_orin.png")
        cv2.imwrite(disted_img_path, final_img)
        return final_img

    def save_depth_map(self, depth_map) -> None:
        depth_map_path = os.path.join(self.save_dir_name, f"depth_map.png")
        normalized_depth = cv2.normalize(
            depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(depth_map_path, colored_depth)

    def save_cloudply_map(self, points=None, color_image=None):
        """
        保存点云为PLY格式

        :param points: 直接的3D点云数据，形状为 (H, W, 3) 或 (B, H, W, 3)
        :param color_image: 彩色图像数据，用于为点云添加颜色信息
        """
        # 处理点云数据
        if len(points.shape) == 4:  # (B, H, W, 3)
            points = points[0]  # 取第一个batch
        H, W, _ = points.shape
        colors = None
        if color_image is not None:
            # 确保颜色图像与点云分辨率匹配
            if color_image.shape[:2] != (H, W):
                color_image = cv2.resize(color_image, (W, H))
            # 将颜色从BGR转换为RGB
            if color_image.shape[2] == 3:
                colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            elif color_image.shape[2] == 4:
                colors = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
            colors = colors.reshape(-1, 3)
        # 生成PLY文件名
        output_dir = self.save_dir_name
        ply_path = os.path.join(output_dir, f"cloud_map.ply")
        # 写入PLY文件
        with open(ply_path, "w") as f:
            # PLY文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            # 计算点的数量
            num_points = H * W
            f.write(f"element vertex {num_points}\n")
            # 顶点属性
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            # 写入点云数据
            points_reshaped = points.reshape(-1, 3)
            for i in range(num_points):
                x, y, z = points_reshaped[i]
                line = f"{x} {y} {z}"

                if colors is not None:
                    r, g, b = colors[i]
                    line += f" {int(r)} {int(g)} {int(b)}"

                f.write(line + "\n")

    def save_depth_np(self, depth_map):
        depth_map_path = os.path.join(self.save_dir_name, f"np_depth.npy")
        np.save(depth_map_path, depth_map)

    def inference_tsr(self):
        pass


# 测试函数
if __name__ == "__main__":
    import os
    import cv2

    # 配置参数
    MODEL_NAME = "Ruicheng/moge-2-vits-normal"
    TEST_IMAGE_PATH = (
        "data/4mm/x9e3mHPUDpzbEHHH4_165742_9401748571828896.jpg"
    )
    OUTPUT_DIR = "output"

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if torch.backends.mps.is_available():
        device_local = "mps"
        print()
    elif torch.cuda.is_available():
        device_local = "cuda:3"
    else:
        device_local = "cpu"

    # 相机内参示例
    test_intrinsics = {
        "fx": 1223.2489013671875,
        "fy": 1217.8367919921875,
        "cx": 971.3410034179688,
        "cy": 533.7720336914062,
        "cawidth": 1920,
        "caheight": 1080,
    }

    dist_coeffs = {
        "k0": -0.2955333888530731,
        "k1": 0.10201843827962875,
        "p1": 0.000291781616397202,
        "p2": -0.0004300259461160749,
        "k2": 0,
    }

    # 1. 初始化模型
    print("初始化模型...")
    try:
        moge_depth = MoGe2_Depth(
            moge2_model=MODEL_NAME,
            image_size=(720, 1280),  # (height, width)
            device=device_local,
        )
        print("✓ 模型初始化成功")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        exit(1)

    # 2. 加载测试图像
    print(f"\n加载测试图像: {TEST_IMAGE_PATH}")
    try:
        img = cv2.imread(TEST_IMAGE_PATH)
        if img is None:
            raise Exception("无法读取图像")

        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"✓ 图像加载成功，尺寸: {img.shape[1]}x{img.shape[0]}")
    except Exception as e:
        print(f"✗ 图像加载失败: {e}")
        exit(1)

    depth_map_frozen = moge_depth.inference_depth(
        img=TEST_IMAGE_PATH,
        frozen_intri=True,
        input_intrinsics=test_intrinsics,
        dist_coeffs=dist_coeffs,
    )
