#############
# author: ice
# 2022.12.12
#############

import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os
import xml.dom.minidom as xdm

import tqdm

os.chdir("/home/lixin/code/MoGe")


def listfiles(rootdir, filelist=[]):
    if os.path.isdir(rootdir):
        subdirs = os.listdir(rootdir)
        for subdir in subdirs:
            subpath = os.path.join(rootdir, subdir)
            filelist = listfiles(subpath, filelist)
    elif os.path.isfile(rootdir):
        filelist.append(rootdir)
    return filelist


def split_jpg_label(filelist, split_format=["jpg", "xml"]):
    all_split_list = []
    for _ in range(len(split_format)):
        all_split_list.append([])
    for filepath in filelist:
        for i in range(len(split_format)):
            if filepath.endswith("." + split_format[i]):
                all_split_list[i].append(filepath)
                break
    return all_split_list


class SignDetector:
    def __init__(self, onnx_path, device_id=0):
        if ort.get_device() == "CPU":
            print("cpu")
            self.ort_session = ort.InferenceSession(onnx_path)
        else:
            print("gpu")
            self.ort_session = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider"],
                provider_options=[{"device_id": device_id}],
            )
        self.input_size = (1024, 1024)
        self.scale = 255
        self.names = [
            "sign_blue_s_roadname",
            "sign_white_n_nomotor",
            "trafficlight",
            "trafficlight_person",
            "sign_yellow_s_electronic_monitoring",
            "sign_blue_s_roadinfo",
            "sign_yellow_s",
            "arrow_l",
            "arrow_ur",
            "electroniceye",
            "sign_white_s_auxiliary",
            "arrow_n_ur",
            "arrow_n_l",
            "sign_yellow_t_crossing",
            "crosswalk_tips",
            "sign_blue_s_crossing",
            "arrow_u",
            "sign_blue_s_public_toilet",
            "sign_green_s_roadinfo",
            "fuzziness",
            "sign_white_n_upspeed_30",
            "sign_blue_n_motorlane",
            "sign_blue_n_bicyclelane",
            "bridge",
            "sign_blue_n_r_lane",
            "sign_red_n_noentry",
            "sign_blue_s_motorlane",
            "sign_blue_n_cararrow_u_motor",
            "sign_green_s_roadinfo_electronic",
            "sign_white_n_upspeed_40",
            "sign_white_n_height_4m",
            "sign_white_n_mixing_limit",
            "sign_white_n_nocar",
            "sign_red_s_pass_lr",
            "sign_blue_n_sailaway",
            "sign_blue_n_cararrow_u_truck",
            "sign_blue_s_roadindication",
            "sign_yellow_s_roadinfo_content",
            "sign_white_n_weight_5t",
            "sign_blue_s",
            "sign_yellow_d_temporary",
            "sign_blue_s_p_motor",
            "sign_yellow_t_temporary",
            "sign_yellow_d_slowly_transit",
            "sign_yellow_t_children",
            "arrow_lu",
            "sign_white_n_noentry_g",
            "arrow_r",
            "sign_white_n_noentry_l_car",
            "sign_blue_n_cararrow_ul_truck",
            "sign_white_n_upspeed_50",
            "sign_white_n_weight_20t",
            "sign_white_n_pedestrianban",
            "sign_white_n_unspeed_30",
            "sign_red_d_stop",
            "sign_blue_n_cararrow_ur_truck",
            "sign_white_n_noentry_r_car",
            "arrow_n_lu",
            "sign_white_n_weight_55t",
            "sign_red_td_rang",
            "sign_white_n_nobicycle",
            "sign_white_s",
            "sign_blue_s_p_bicycle",
            "sign_blue_s_detour",
            "sign_green_s_roadinfo_roadsuperhighway",
            "sign_yellow_t_confluent_r",
            "sign_green_s_indication",
            "sign_green_s_roadinfo_exit",
            "sign_white_n_upspeed_80",
            "sign_green_s",
            "sign_green_s_roadindication",
            "sign_white_s_roadinfo_superhighway",
            "sign_white_n_upspeed_60",
            "sign_blue_s_roadinfo_cartruck",
            "sign_blue_s_roadinfo_car",
            "sign_white_n_weight_49t",
            "sign_white_n_weightr_14t",
            "sign_blue_n_l_lane",
            "sign_blue_s_r_lane",
            "sign_white_n_weight_40t",
            "sign_blue_n_cararrow_l_truck",
            "arrow_n_u",
            "sign_yellow_s_roadinfo",
            "sign_white_n_height_4.3m",
            "arrow_g",
            "sign_white_n_height_2.2m",
            "sign_white_n_nodangercar",
            "sign_white_n_weightr_13t",
            "sign_blue_s_pass_lr",
            "sign_yellow_s_slowdown",
            "sign_blue_n_instruct_r",
            "sign_blue_s_divert_r",
            "arrow_45l",
            "sign_blue_n_nostop",
            "sign_blue_n_cararrow_r_bicycle",
            "sign_white_n_weight_30t",
            "arrow_lg",
            "sign_white_n_nohonking",
            "sign_blue_n_person",
            "sign_yellow_t_intersection_t_right",
            "sign_blue_s_electronic_monitoring",
            "arrow_g_y",
            "sign_yellow_t_danger",
            "sign_red_s_road",
            "sign_blue_s_roadinfo_content",
            "sign_green_s_road",
            "arrow_n_r",
            "sign_white_n_bicyclelane_motor_dotted_line",
            "sign_blue_n_p_bicycle",
            "sign_white_s_provincial_lr",
            "sign_white_s_roadinfo_content",
            "sign_white_n_height_4.8m",
            "sign_white_n_turnround_l",
            "sign_white_n_upspeed_5",
            "sign_yellow_t_tidallane",
            "sign_blue_arrow_roadinfo",
            "sign_white_n_noentry_lr",
            "sign_red_s_travel",
            "sign_blue_n_cararrow_ur_motor",
            "sign_yellow_t_slowly_transit",
            "sign_red_s_pass_r",
            "sign_blue_n_cararrow_ul_motor",
            "sign_green_s_divert_l",
            "sign_white_n_nopass",
            "sign_blue_s_sailaway",
            "sign_black_s_roadinfo_electronic",
            "sign_blue_s_p",
            "sign_white_n_noentry_g_truck",
            "sign_white_n_noentry_ur_truck",
            "sign_blue_s_pedestrian_bridge",
            "sign_yellow_s_roadindication",
            "sign_yellow_s_bicyclelane_person",
            "sign_white_n_height_3m",
            "sign_white_n_height_3.2m",
            "sign_blue_n_cararrow_u_motorcycle",
            "sign_blue_n_cararrow_lr_motor",
            "sign_blue_s_instruct_r",
            "sign_white_s_p",
            "sign_white_n_noentry_r_truck",
            "sign_white_n_noentry_l",
            "sign_yellow_s_detour",
            "sign_blue_n_cararrow_u_bicycle",
            "sign_blue_n_p",
            "arrow_ug",
            "sign_blue_s_inducedarrow",
            "sign_yellow_t_lowhill",
            "sign_yellow_t_narrow_r",
            "sign_white_s_provincial_r",
            "sign_yellow_t_narrow_lr",
            "sign_blue_s_dislocation_track",
            "sign_yellow_t_obstructions_l",
            "sign_white_s_railwayway",
            "sign_yellow_t_overwater",
            "sign_green_d_confluent_r",
            "sign_yellow_t_narrow_bridge",
            "sign_white_td_rang",
            "sign_yellow_t_accident prone",
            "sign_yellow_t_intersection_crossroads",
            "sign_yellow_t_intersection_t_left",
            "sign_yellow_t_crosswind",
            "sign_yellow_t_falling rocks_r",
            "sign_yellow_t_narrow_l",
            "sign_white_s_provincial_tr",
            "sign_yellow_t_railwaycrossing",
            "arrow_lur",
            "arrow_n_lur",
            "sign_white_n_upspeed_20",
            "sign_yellow_t_tunnellights",
            "arrow_n_lr",
            "sign_yellow_t_slippery",
            "sign_white_s_provincial_l",
            "sign_white_s_provincial_u",
            "sign_white_t_railwaycrossing_2",
            "sign_yellow_t_embankment_r",
            "sign_white_t_blackspot",
            "sign_white_t_railwaycrossing_3",
            "sign_white_s_roadname",
            "sign_yellow_t_obstructions_lr",
            "sign_blue_s_disabled",
            "sign_white_n_height_4.5m",
            "sign_yellow_t_turn_r",
            "sign_white_s_electronic_monitoring",
            "sign_yellow_s_fuzziness",
            "sign_blue_n_cararrow_lr_truck",
            "arrow_no",
            "sign_green_s_roadinfo_car",
            "sign_blue_n_downspeed_70",
            "sign_blue_n_downspeed_60",
            "sign_blue_n_downspeed_50",
            "sign_green_s_roadinfo_cartruck",
            "sign_green_s_dislocation_track",
            "sign_green_d_milestones",
            "sign_red_s_divert_r",
            "sign_red_s_divert_l",
            "arrow_45r",
            "sign_white_n_height_4.2m",
            "sign_blue_n_cararrow_ur_car",
            "sign_blue_n_cararrow_u_car",
            "sign_white_s_electroniceye",
            "sign_blue_s_oneway_l",
            "sign_blue_n_instruct_u",
            "sign_blue_n_cararrow_ul_car",
            "sign_blue_n_pass_lr",
            "sign_blue_s_deadend",
            "arrow_n_g_y",
            "sign_blue_n_cararrow_lr_bicycle",
            "sign_blue_n_cararrow_ur_bicycle",
            "sign_blue_s_oneway_u",
            "sign_white_n_noentry_l_bicycle",
            "sign_white_n_noentry_l_truck",
            "sign_blue_n_cararrow_r_motor",
            "sign_white_n_upspeed_35",
            "sign_blue_n_instruct_l",
            "sign_green_s_divert_r",
            "sign_blue_n_cararrow_r_motorcycle",
            "sign_blue_s_deadend_bicycle",
            "sign_blue_s_bicyclelane",
            "sign_red_s_detour",
            "arrow_lr",
            "sign_blue_s_cararrow_lr_motor",
            "sign_white_n_height_3.8m",
            "sign_white_n_p",
            "sign_white_n_height_2.0m",
            "sign_blue_s_divert_l",
            "sign_white_n_height_2.4m",
            "sign_blue_n_deadend_car",
            "sign_white_n_height_3.5m",
            "sign_white_n_height_4.0m",
            "sign_green_s_roadinfo_tollbooths",
            "sign_blue_n_cararrow_ur_motorcycle",
            "sign_white_n_motorlane",
            "arrow_n_g",
            "sign_green_s_roadinfo_distance",
            "sign_white_n_noentry_r",
            "sign_white_n_noentry_u_truck",
            "sign_white_s_roadinfo",
            "sign_yellow_n_slowly_transit",
            "sign_yellow_s_divert_l",
            "sign_yellow_n_bicyclelane_person",
            "sign_yellow_s_bicyclelane",
            "sign_white_n_noparking",
            "sign_blue_s_turnround_l",
            "sign_blue_s_indication",
            "sign_blue_n_cararrow_lr_car",
            "sign_green_s_roadinfo_service",
            "sign_green_s_roadinfo_superhighway",
            "sign_green_s_roadinfo_entersuperhighway",
            "sign_green_s_roadinfo_content",
            "sign_yellow_t_turn_l",
            "arrow_n_gr",
            "sign_yellow_s_oneway_l",
            "sign_white_n_height_1.8m",
            "sign_white_n_p_bicycle",
            "sign_blue_n_cararrow_ul_bicycle",
            "sign_white_s_public_toilet",
            "sign_white_n_weight_15t",
            "sign_white_n_upspeed_55",
            "sign_yellow_t_reverse_turn",
            "sign_white_n_height_5m",
            "sign_yellow_s_roadinfo_tollbooths",
            "sign_green_s_roadinfo_emergency",
            "sign_white_n_upspeed_120",
            "sign_blue_n_downspeed_110",
            "sign_blue_n_downspeed_90",
            "sign_green_s_roadname",
            "sign_blue_d_confluent_r",
            "sign_white_n_weightr_10t",
            "sign_red_s_slowdown",
            "sign_white_n_weight_10t",
            "sign_white_n_noentry_g_car",
            "sign_blue_s_oneway_r",
            "sign_white_n_slowly_transit",
            "sign_blue_n_bicyclelane_person",
            "sign_white_n_height_6.2m",
            "arrow_n_45l",
            "sign_blue_s_roadinfo_electronic",
            "sign_red_arrow_roadinfo",
            "sign_white_n_noentry_u_car",
            "sign_white_n_height_5.2m",
            "sign_blue_n_cararrow_l_bicycle",
            "sign_yellow_t_twoway",
            "sign_white_n_unspeed_40",
            "sign_white_t_electroniceye",
            "sign_white_n_noentry_u",
            "sign_black_s_roadinfo",
            "sign_blue_s_roadinfo_superhighway",
            "sign_white_n_height_2.5m",
            "sign_green_s_electroniceye",
            "sign_blue_n_downspeed_80",
            "sign_white_n_upspeed_100",
            "sign_blue_n_instruct_lr",
            "sign_yellow_s_divert_r",
            "sign_white_n_height_2m",
            "sign_blue_d_milestones",
            "sign_blue_n_turnround_l",
            "sign_white_n_noentry",
            "sign_blue_s_roadinfo_motor",
            "sign_white_n_height_5.5m",
            "sign_blue_n_instruct_ur",
            "arrow_special",
            "sign_green_s_p",
            "sign_blue_s_lr_lane",
            "sign_white_n_upspeed_15",
            "sign_green_n_p_motor",
            "sign_green_s_electronic_monitoring",
            "sign_white_s_slowdown",
            "sign_white_arrow_roadinfo",
            "sign_blue_n_lr_lane",
            "sign_blue_d_temporary",
            "sign_yellow_s_electroniceye",
            "sign_blue_d_shunt_r",
            "sign_white_s_road",
            "sign_green_n_lr_lane",
            "sign_white_n_upspeed_70",
            "sign_green_s_roadinfo_motor",
            "sign_white_n_noovertaking",
            "sign_yellow_t_confluent_l",
            "sign_yellow_t_intersection_y_merge",
            "sign_white_n_height_5.0m",
            "sign_white_t_intersection_t_right",
            "sign_green_n_p",
            "sign_blue_n_instruct_lur",
            "arrow_n_ug",
            "sign_yellow_t_lowlyingroad",
            "sign_whtie_s_roadname",
            "sign_yellow_t_mountain_l",
            "sign_yellow_d_shunt_r",
            "sign_yellow_t_mountain_r",
            "sign_blue_n_turnround_r",
            "sign_white_n_weight_25t",
            "_yellow_t_slowly_transit",
            "sign_blue_s_slowdown",
            "sign_yellow_t_ferries",
            "sign_green_d_confluent_l",
            "sign_yellow_t_roundabout",
            "sign_blue_d_shunt_l",
            "sign_yellow_t_falling rocks_l",
            "sign_yellow_t_railwayperson",
            "sign_yellow_t_humpbridge",
            "sign_blue_n_roundabout",
            "sign_blue_s_roadinfo_exit",
            "sign_white_s_roadinfo_distance",
            "sign_white_n_unspeed_60",
            "sign_yellow_s_pass_lr",
            "sign_yellow_s_upspeed_20",
            "sign_green_s_roadinfo_truck",
            "sign_green_d_shunt_r",
            "sign_red_s",
            "sign_yellow_s_person",
            "sign_green_s_lr_lane",
            "sign_red_n_pass_l",
            "sign_red_n_pass_r",
            "sign_white_n_nolong_stop",
            "arrow_n_ul",
            "sign_yellow_t_intersection_right",
            "sign_yellow_s_roadinfo_distance",
            "sign_green_s_bicyclelane",
            "sign_white_d_shunt_r",
            "sign_white_n_notwocar",
            "sign_yellow_s_upspeed_80",
            "sign_yellow_n_upspeed_40",
            "sign_blue_s_l_lane",
            "sign_white_t_children",
            "sign_yellow_s__roadinfo_distance",
            "sign_green_s_motorlane_cartruck",
            "sign_green_s_motorlane",
            "sign_blue_n_flyover",
            "sign_yellow_t_intersection_y",
            "sign_white_s_arrow_roadinfo",
            "sign_white_n_width_3.5m",
            "sign_yellow_t_embankment_l",
            "sign_yellow_t_bumpyroad",
            "sign_yellow_t_roughroad",
            "sign_yellow_t_livestock",
            "sign_white_n_weightr_7t",
            "sign_white_n_danger_bridge",
            "sign_blue_d_confluent_l",
            "sign_white_t_danger",
            "sign_white_n_chinese",
            "sign_yellow_s_auxiliary",
            "sign_blue_s_signage",
            "sign_yellow_t_villages",
            "sign_yellow_t_intersection",
            "sign_white_n_nobicycle_downhill",
            "sign_white_s_detour",
            "sign_white_s_railwayperson",
            "sign_red_n_upspeed_5",
            "sign_blue_s_electroniceye",
            "sign_red_s_roadinfo",
            "sign_yellow_t_intersection_t",
            "sign_yellow_s_road",
            "sign_yellow_t_shunt_l",
            "sign_blue_n_instruct_ul",
            "sign_red_t_accident prone",
            "sign_white_s_roadinfo_exit",
            "sign_yellow_n_nomotor",
            "sign_yellow_s_children",
            "sign_white_n_unspeed_50",
            "sign_blue_n_crossing",
            "sign_yellow_t_electroniceye",
            "sign_white_n_accident prone",
            "sign_yellow_t_bicyclelane",
            "sign_white_n_weight_50t",
            "sign_red_s_p",
            "sign_yellow_td_intersection_t_right",
            "sign_yellow_td_intersection_crossroads",
            "sign_yellow_td_intersection_t",
            "sign_yellow_td_intersection_t_left",
            "sign_white_n_noentry_lr_truck",
            "arrow_n_lg",
            "sign_blue_n_cararrow_lr_motorcycle",
            "sign_blue_s_roadinfo_largecar",
            "sign_yellow_s_crossing",
            "sign_blue_s_roadinfo_distance",
            "sign_yellow_s_electronic",
            "r",
            "sign_white_s_indication",
            "sign_green_s_roadinfo_largecar",
            "sign_white_n_width_3m",
            "sign_yellow_s_upspeed_5",
            "sign_white_s_roadinfo_car",
            "sign_white_s_roadinfo_cartruck",
            "sign_green_s_slowdown",
            "sign_black_n_upspeed_60",
            "sign_green_s_confluent_r",
            "sign_red_s_pass_l",
            "sign_white _s_tunnellights",
            "sign_blue_s_tunnellights",
            "sign_blue_n_p_motor",
            "sign_blue_n_cararrow_l_motorcycle",
            "sign_white_n_height_2.1m",
            "sign_red_s_slowly_transit",
        ]

    def preprocess(self, img):
        img = cv2.resize(img, self.input_size)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        img = img / self.scale
        img = img[None]
        return img

    def detect(self, img, conf_thres=0.25, iou_thres=0.45, max_det=300):
        img = img.astype(np.float32)
        h, w, _ = img.shape
        input_data = self.preprocess(img)
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}

        ort_outs = self.ort_session.run(None, ort_inputs)[0]
        outputs = self.non_max_suppression(ort_outs, conf_thres, iou_thres, max_det)[0]
        for output in outputs:
            output[0] = max(0, min(w - 1, int(output[0] * w / self.input_size[0])))
            output[1] = max(0, min(h - 1, int(output[1] * h / self.input_size[1])))
            output[2] = max(0, min(w - 1, int(output[2] * w / self.input_size[0])))
            output[3] = max(0, min(h - 1, int(output[3] * h / self.input_size[1])))
        return outputs

    def matrix_iou(self, box1, box2, eps=1e-7):
        """
        :param box1: (n, 4)
        :param box2: (m, 4)
        """
        lt = np.maximum(box1[:, None, :2], box2[:, :2])
        rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
        wh = np.maximum(rb - lt + 1, 0)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]
        box1_areas = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
        box2_areas = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)
        iou_matrix = inter_areas / (box1_areas[:, None] + box2_areas - inter_areas + eps)
        return iou_matrix

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 1024  # (pixels) maximum box width and height
        max_nms = 3000  # maximum number of boxes into torchvision.ops.nms()
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

        output = [np.zeros((0, 6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(
                x[:, :4]
            )  # center_x, center_y, width, height) to (x1, y1, x2, y2)

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero()
                x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].astype(np.float64)), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = np.concatenate((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort()[::-1]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            indexes = np.array(list(range(boxes.shape[0])))
            keep = []
            while len(indexes) > 0:
                i = indexes[0]
                keep.append(i)
                # cls_idxs = int(boxes[0][0] // max_wh) * max_wh <= boxes[1:][0] < int(boxes[0][0] // max_wh + 1) * max_wh
                # cls_boxes = boxes[cls_idxs]
                iou_matrix = self.matrix_iou(boxes[i][None], boxes[indexes[1:]])
                inds = np.where((iou_matrix < iou_thres).squeeze())[0]
                indexes = indexes[inds + 1]
            # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            keep = np.array(keep)
            if keep.shape[0] > max_det:  # limit detections
                keep = keep[:max_det]

            output[xi] = x[keep]

        return output

    def draw(self, img, output):
        for out in output:
            out = list(out)
            cv2.rectangle(
                img, (int(out[0]), int(out[1])), (int(out[2]), int(out[3])), (0, 0, 255), 1
            )
            cv2.putText(
                img,
                self.names[int(out[5])],
                (int(out[0]), int(out[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
            )
        return img

    def save_image(self, img, img_path):
        cv2.imwrite(img_path, img)

    def save_xml(self, img, xml_path, output):
        h, w, c = img.shape
        filename = xml_path.split(os.sep)[-1]
        folder = xml_path.strip(filename)
        labeldata = {}
        labeldata["folder"] = folder
        labeldata["filename"] = filename
        labeldata["width"] = w
        labeldata["height"] = h
        labeldata["channels"] = c
        objs = []
        for out in output:
            obj = {}
            obj["xmin"] = float(out[0])
            obj["ymin"] = float(out[1])
            obj["xmax"] = float(out[2])
            obj["ymax"] = float(out[3])
            obj["score"] = float(out[4])
            obj["name"] = self.names[int(out[5])]
            if obj["name"] == "r":
                obj["name"] = "arrow_n_r"
            obj["difficult"] = 0
            objs.append(obj)
        labeldata["objs"] = objs
        self.generate_voc_xml(xml_path, labeldata)

    # Write voc label
    def generate_voc_xml(self, anno_path, labeldata):
        w = labeldata["width"]
        h = labeldata["height"]
        c = labeldata["channels"]
        rects = labeldata["objs"]
        xml = xdm.Document()
        annotation = xml.createElement("annotation")
        xml.appendChild(annotation)
        folder = xml.createElement("folder")
        folder_value = xml.createTextNode(labeldata["folder"])
        folder.appendChild(folder_value)
        filename = xml.createElement("filename")
        filename_value = xml.createTextNode(labeldata["filename"])
        filename.appendChild(filename_value)
        size = xml.createElement("size")
        width = xml.createElement("width")
        height = xml.createElement("height")
        depth = xml.createElement("depth")
        width_value = xml.createTextNode(str(w))
        height_value = xml.createTextNode(str(h))
        depth_value = xml.createTextNode(str(c))
        width.appendChild(width_value)
        height.appendChild(height_value)
        depth.appendChild(depth_value)
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        annotation.appendChild(folder)
        annotation.appendChild(filename)
        annotation.appendChild(size)
        for rect in rects:
            left = round(rect["xmin"])
            top = round(rect["ymin"])
            right = round(rect["xmax"])
            bottom = round(rect["ymax"])

            object = xml.createElement("object")
            name = xml.createElement("name")
            score = xml.createElement("score")
            bndbox = xml.createElement("bndbox")
            xmin = xml.createElement("xmin")
            ymin = xml.createElement("ymin")
            xmax = xml.createElement("xmax")
            ymax = xml.createElement("ymax")
            name_value = xml.createTextNode(rect["name"])
            score_value = xml.createTextNode(str(rect["score"]))
            xmin_value = xml.createTextNode(str(left))
            ymin_value = xml.createTextNode(str(top))
            xmax_value = xml.createTextNode(str(right))
            ymax_value = xml.createTextNode(str(bottom))
            name.appendChild(name_value)
            score.appendChild(score_value)
            xmin.appendChild(xmin_value)
            ymin.appendChild(ymin_value)
            xmax.appendChild(xmax_value)
            ymax.appendChild(ymax_value)
            bndbox.appendChild(xmin)
            bndbox.appendChild(ymin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymax)
            object.appendChild(name)
            object.appendChild(bndbox)
            annotation.appendChild(object)
        with open(anno_path, "wb") as xmlfile:
            # xml.writexml(xmlfile, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
            xmlfile.write(xml.toprettyxml(indent="\t", encoding="utf-8"))


def main(args):
    # sign_detector_path = 'runs/train/org_sign/weights/best.onnx'
    # 检测模型
    sign_detector = SignDetector(args.weights)

    # 检测视频
    if args.video != "":
        video_pth = args.video
        cap = cv2.VideoCapture(video_pth)
        if args.show:
            cv2.namedWindow("sign", cv2.WINDOW_NORMAL)
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
        save_num = 0
        frame_num = 0
        while 1:
            try:
                ret, frame = cap.read()
            except:
                break
            if not ret or frame is None:
                break
            print(frame_num)
            results = sign_detector.detect(frame)
            if args.save_xml and args.save_dir and frame_num % args.frame_interval == 0:
                img_path = os.path.join(args.save_dir, str(save_num) + ".jpg")
                sign_detector.save_image(frame, img_path)
                xml_path = os.path.join(args.save_dir, str(save_num) + ".xml")
                sign_detector.save_xml(frame, xml_path, results)
                save_num += 1
            if args.show:
                frame = sign_detector.draw(frame, results)
                cv2.imshow("sign", frame)
                if cv2.waitKey(1) == 27:
                    cv2.destroyWindow("sign")
                    break
            frame_num += 1
    #检测单张图像
    elif args.img != "":
        save_num = 0
        img = cv2.imread(args.img)
        results = sign_detector.detect(img)
        if args.save_xml and args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            if args.save_img:
                img_path = os.path.join(args.save_dir, args.img.split(os.sep)[-1])
                sign_detector.save_image(img, img_path)
            xml_path = os.path.join(args.save_dir, str(save_num) + ".xml")
            sign_detector.save_xml(img, xml_path, results)
            save_num += 1
        if args.show:
            img = sign_detector.draw(img, results)
            cv2.namedWindow("sign", cv2.WINDOW_NORMAL)
            cv2.imshow("sign", img)
            cv2.waitKey(0)
    #检测图像文件夹
    elif args.imgs_dir != "":
        listfile = listfiles(args.imgs_dir)
        if args.show:
            cv2.namedWindow("sign", cv2.WINDOW_NORMAL)
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
        for file in tqdm.tqdm(listfile, ncols=0):
            # print(file)
            if not (file.endswith(".jpg") or file.endswith(".png")):
                continue
            img = cv2.imread(file)
            results = sign_detector.detect(img)
            if args.save_xml and args.save_dir:
                if args.save_img:
                    img_path = os.path.join(args.save_dir, file.split(os.sep)[-1])
                    sign_detector.save_image(img, img_path)
                if file.endswith(".jpg"):
                    xml_path = os.path.join(
                        args.save_dir, file.split(os.sep)[-1].replace(".jpg", ".xml")
                    )
                    sign_detector.save_xml(img, xml_path, results)
                if file.endswith(".png"):
                    xml_path = os.path.join(
                        args.save_dir, file.split(os.sep)[-1].replace(".png", ".xml")
                    )
                    sign_detector.save_xml(img, xml_path, results)
            if args.show:
                img = sign_detector.draw(img, results)
                cv2.imshow("sign", img)
                if cv2.waitKey(0) == 27:
                    cv2.destroyWindow("sign")
                    break
    #检测文件的列表（就是某个文件夹，传入一些图片文件名的列表（部分图像））
    elif args.imgfile != "":
        fid = open(args.imgfile, "r")
        listfile = fid.readlines()
        if args.show:
            cv2.namedWindow("sign", cv2.WINDOW_NORMAL)
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
        for file in listfile:
            print(file)
            img = cv2.imread(file.strip().strip("\n").split(".jpg")[0] + ".jpg")
            results = sign_detector.detect(img)
            if args.save_xml and args.save_dir:
                if args.save_img:
                    img_path = os.path.join(args.save_dir, file.split(os.sep)[-1])
                    sign_detector.save_image(img, img_path)
                if file.endswith(".jpg"):
                    xml_path = os.path.join(
                        args.save_dir, file.split(os.sep)[-1].replace(".jpg", ".xml")
                    )
                    sign_detector.save_xml(img, xml_path, results)
                if file.endswith(".png"):
                    xml_path = os.path.join(
                        args.save_dir, file.split(os.sep)[-1].replace(".png", ".xml")
                    )
                    sign_detector.save_xml(img, xml_path, results)
            if args.show:
                img = sign_detector.draw(img, results)
                cv2.imshow("sign", img)
                if cv2.waitKey(0) == 27:
                    cv2.destroyWindow("sign")
                    break
        fid.close()


#  python sign_detector.py --weights onnxpath --imgs-dir imgpath  --save-xml --save-img --save-dir savepath
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignNet")
    parser.add_argument("--img", default="", type=str, metavar="PATH", help="imgpath")
    parser.add_argument(
        "--imgs-dir", default="dataset/cdsz/test/", type=str, metavar="PATH", help="imgsdir"
    )
    parser.add_argument("--imgfile", default="", type=str, metavar="PATH", help="imgfile")
    parser.add_argument("--video", default="", type=str, metavar="PATH", help="video")
    parser.add_argument(
        "--weights",
        default="weights_detect/sign_detector.onnx",
        type=str,
        metavar="PATH",
        help="weights",
    )
    parser.add_argument("--show", action="store_true", help="show image")
    parser.add_argument("--save-xml", action="store_true", help="save xml labels")
    parser.add_argument("--save-img", action="store_true", help="save image")
    parser.add_argument(
        "--save-dir", default="output/detect", type=str, metavar="PATH", help="save dir"
    )
    parser.add_argument(
        "--frame-interval", default=30, type=int, metavar="N", help="frame interval"
    )
    main(parser.parse_args())
