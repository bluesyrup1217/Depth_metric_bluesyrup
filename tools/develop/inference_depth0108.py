####################################
####
####Code editing: Xin Li (Steve)
####Edit time: 2026-01-06
####
####################################
import sys
import json

sys.path.append("./")
import cv2
import torch
import numpy as np
import os
import math
import time
from moge.model.v2 import MoGeModel


class MoGe2_Depth:
    def __init__(self, moge2_model: str, image_size=(720, 1280), device="cuda:0"):
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
        img_infer_time:int = 0
    ):
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
        if save_visual_result and img_infer_time == 0:
            saveresult_time_begin = time.time()
            self.save_np(
                depth_map=depth_map, save_type="depth"
            )  # 保存深度图为numpy数组
            self.save_np(
                depth_map=points, save_type="points"
            )  # 这里复用一下保存点云为np数组
            self.save_depth_map(depth_map=depth_map)  # 保存深度图
            self.save_cloudply_map(
                points=points, color_image=img_removal_dist
            )  # 保存点云为PLY格式
            saveresult_time_end = time.time()
            print(
                f"保存推理结果时间: {saveresult_time_end - saveresult_time_begin:.2f}s"
            )
        else:
            # print("Non save result image!")
            pass

        return depth_map, points
        # points的某一像素的值包含一个xyz的坐标，但是xy的值是基于光心的偏移，左半部分图像的x坐标为负，上半部分图像的y坐标为负

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

    def save_np(self, depth_map, save_type="depth"):
        if save_type == "depth":
            depth_map_path = os.path.join(self.save_dir_name, f"np_depth.npy")
        elif save_type == "points":
            depth_map_path = os.path.join(self.save_dir_name, f"np_points.npy")
        np.save(depth_map_path, depth_map)

    def read_json_box(self, json_box_path):
        """
        读取JSON文件并提取每个对象的xywh和classid信息

        :param json_path: JSON文件路径
        :return: 包含每个检测框信息的列表，每个元素为{"x": x, "y": y, "w": w, "h": h, "classid": classid}
        """
        try:
            with open(json_box_path, "r") as f:
                data = json.load(f)

            # 初始化结果列表
            result_list = []

            # 处理不同的JSON结构
            if isinstance(data, list):
                # 直接是对象列表
                boxes_data = data
            elif isinstance(data, dict):
                # 可能包含boxes或其他字段
                if "boxes" in data:
                    boxes_data = data["boxes"]
                elif "detections" in data:
                    boxes_data = data["detections"]
                else:
                    # 尝试将字典作为单个对象处理
                    boxes_data = [data]
            else:
                raise ValueError("JSON数据格式不支持")

            # 提取每个对象的信息
            for item in boxes_data:
                if isinstance(item, dict):
                    # 提取xywh和classid
                    x = item.get("x", 0)
                    y = item.get("y", 0)
                    # 处理可能的width/height或w/h命名
                    w = item.get("width", item.get("w", 0))
                    h = item.get("height", item.get("h", 0))
                    classid = item.get(
                        "classid", item.get("class_id", item.get("class", -1))
                    )

                    # 添加到结果列表
                    result_list.append((x, y))

            return result_list, len(result_list)

        except FileNotFoundError:
            print(f"错误: JSON文件 {json_box_path} 不存在")
            return []
        except json.JSONDecodeError:
            print(f"错误: JSON文件 {json_box_path} 格式不正确")
            return []
        except Exception as e:
            print(f"处理JSON文件时发生错误: {e}")
            return []

    def read_json_config(self, json_config_path):
        """
        读取JSON文件中的所有变量并返回字典

        :param json_path: JSON文件路径
        :return: 包含所有变量的字典
        """
        try:
            with open(json_config_path, "r") as f:
                data = json.load(f)
            # 直接返回解析后的所有数据
            return data

        except FileNotFoundError:
            print(f"错误: JSON文件 {json_config_path} 不存在")
            return {}
        except json.JSONDecodeError:
            print(f"错误: JSON文件 {json_config_path} 格式不正确")
            return {}
        except Exception as e:
            print(f"处理JSON文件时发生错误: {e}")
            return {}

    def infer_coordinate_value(self, json_box_path, json_config_path):
        self.inference_id = time.strftime("%Y%m%d_%H:%M:%S")
        self.save_dir_name = f"output/{self.inference_id}"
        os.makedirs(self.save_dir_name, exist_ok=True)
        box_list, box_num = self.read_json_box(json_box_path)
        config_dict = self.read_json_config(json_config_path)
        image_path = config_dict["image_path"]
        image_intrinsics = config_dict["input_intrinsics"]
        image_dist_coeffs = config_dict["dist_coeffs"]
        frozen_intri = config_dict["frozen_intri"]
        save_visual_result = config_dict["save_visual_result"]
        res = []
        for i, box in enumerate(box_list):
            x, y = box
            depth_map, points = self.inference_depth(
                img=image_path,
                frozen_intri=frozen_intri,
                input_intrinsics=image_intrinsics,
                dist_coeffs=image_dist_coeffs,
                save_visual_result=save_visual_result,
                img_infer_time=i,
            )
            try:
                u, v, z = points[y][x][0], points[y][x][1], points[y][x][2]
            except:
                u, v, z = 5.0, 2.0, 0.0
            # 将float32类型转换为Python的float类型以便JSON序列化
            res.append({"u": float(u), "v": float(v), "z": float(z)})
        try:
            with open(os.path.join(self.save_dir_name, f"inference_coordinate.json"), "w") as f:
                json.dump(res, f, indent=4)
        except Exception as e:
            print(f"保存JSON文件时发生错误: {e}")
            
            
        
        pass


# 测试函数
if __name__ == "__main__":

    # init 参数
    MODEL_NAME = "Ruicheng/moge-2-vits-normal"
    TEST_IMAGE_PATH = "data/4mm/x9e3mHPUDpzbEHHH4_165742_9401748571828896.jpg"
    IMAGE_SIZE=(720, 1280)

    if torch.backends.mps.is_available():
        DEVICE_LOCAL = "mps"
        print("使用mps设备!")
    elif torch.cuda.is_available():
        DEVICE_LOCAL = "cuda:3"
        print("使用cuda:3设备!")
    else:
        DEVICE_LOCAL = "cpu"
        print("使用cpu设备!")

    moge_depth = MoGe2_Depth(
        moge2_model=MODEL_NAME,
        image_size=IMAGE_SIZE,
        device=DEVICE_LOCAL,
    )
        
    moge_depth.infer_coordinate_value(
        json_box_path="inference_box.json",
        json_config_path="inference_params.json",
    )
