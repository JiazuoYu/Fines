import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk
from tqdm import tqdm
import pdb
import os
from PIL import Image as PILImage
import re
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2
import pycocotools
from loguru import logger
from PIL import Image
import ast
import difflib
import time



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--cascade_reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--segmentation_config_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--test_json_path", type=str, default="the path of the json file of the test data")
    parser.add_argument('--resize_size', type=str, default="(1920, 1080)",help='Resize image to a tuple string like "(1920, 1080)"')
    parser.add_argument('--cascade_resize_size', type=str, default="(512, 512)",help='Resize cascade image to a tuple string like "(512, 512)"')
    parser.add_argument("--image_path", type=str, default="image data path")
    parser.add_argument("--save_results", action="store_true", default=False)
    parser.add_argument("--dynamic_box", action="store_true", default=False)
    parser.add_argument("--save_path", default="PATH_TO_SAVE_FILE", type=str)
    parser.add_argument(
        "--qa_stage",
        default="stage2",
        type=str,
        choices=["stage1", "stage2", "no_qa"],
        help="precision for inference",
    )
    return parser.parse_args()

def is_inside(small_box, large_box):
    sx_min, sy_min, sx_max, sy_max = small_box
    lx_min, ly_min, lx_max, ly_max = large_box

    return (
        sx_min >= lx_min and
        sy_min >= ly_min and
        sx_max <= lx_max and
        sy_max <= ly_max
    )

def get_bbox_from_mask(mask, width, height):
    """
    通过二值 mask 计算外接矩形框 (bbox)。
    :param mask: 二值化 mask (numpy array)
    :return: (x_min, y_min, x_max, y_max) - 外接矩形框坐标
    """
    # logger.info("ori size:{}, new size:{}".format((height, width), new_size))
    
    # 找到 mask 中所有非零点的坐标
    y_coords, x_coords = np.nonzero(mask)  
    
    x_min = x_coords.min()  
    x_max = x_coords.max()  
    y_min = y_coords.min()  
    y_max = y_coords.max()

    
    return (x_min, y_min, x_max, y_max)

def extract_bbox_points_think(output_text, x_factor, y_factor, is_final_answer_model=False):
    json_pattern = r'{[^}]+}'  # 匹配最简单的JSON对象
    json_match = re.search(json_pattern, output_text)

    points = None # 防止region不预测point报错
    # pdb.set_trace()
    if json_match:
        # print(json_match.group(0).replace(']"', ']')) # replace some bad char
        json_str = json_match.group(0).replace(']"', ']')
        # 尝试解析 JSON，如果失败则认为没有匹配到有效内容
        data = None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 处理单引号的情况：使用 ast.literal_eval 解析 Python 字典格式
            try:
                data = ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                # 如果解析失败，视为没有匹配到有效内容
                data = None
        
        if data is not None:
            final_response = ""  # 初始化 final_response
            if is_final_answer_model:
                try:
                    final_response = data["response"]
                except:
                    final_response = ""
                    logger.info("final_response is None!!!")
            # assert 1 == 2
            # 查找bbox键
            bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
            # pdb.set_trace()
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                content_bbox = [round(int(content_bbox[0])*x_factor), round(int(content_bbox[1])*y_factor), round(int(content_bbox[2])*x_factor), round(int(content_bbox[3])*y_factor)]
            else:
                content_bbox = None 
            # 查找points键
            points_keys = [key for key in data.keys() if 'points' in key.lower()][:2]  # 获取前两个points键
            if len(points_keys) == 2:
                point1 = data[points_keys[0]]
                point2 = data[points_keys[1]]
                point1 = [round(int(point1[0])*x_factor), round(int(point1[1])*y_factor)]
                point2 = [round(int(point2[0])*x_factor), round(int(point2[1])*y_factor)]
                points = [point1, point2]
        else:
            # 如果解析失败，视为没有匹配到有效内容
            content_bbox = None
            points = None
            final_response = ""
    else:
        content_bbox = None
        points = None
        final_response = ""
        logger.info("No response, box and points matched !!!")
        
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)
        # logger.info("Thinking process:{}".format(think_text))
    else:
        think_text = None
        logger.info("No thinking process matched !!!")
    if is_final_answer_model:
        return content_bbox, points, think_text, final_response
    return content_bbox, points, think_text

def load_data_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    annotations = data["annotations"]
    
    return annotations

def get_mask_from_points(anno, img):
    
    height, width = img.shape[:2]
    points = anno["points"]
    label_value = 1  # target

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
    cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)
    
    return mask

def intersectionAndUnionCPU(output, target, K, ignore_index=255):
    """
    Compute intersection and union for segmentation evaluation on CPU.
    
    Args:
        output (numpy.ndarray): Predicted segmentation mask, shape (H, W), values in range [0, K-1].
        target (numpy.ndarray): Ground truth segmentation mask, shape (H, W), values in range [0, K-1].
        K (int): Number of classes.
        ignore_index (int, optional): Label to ignore in evaluation. Default is 255.

    Returns:
        area_intersection (numpy.ndarray): Per-class intersection count.
        area_union (numpy.ndarray): Per-class union count.
        area_target (numpy.ndarray): Per-class ground truth count.
    """
    assert output.shape == target.shape, "output and target must have the same shape"
    
    mask = target != ignore_index  # Mask to ignore ignore_index
    output = output[mask]
    target = target[mask]
    
    intersection = output[output == target]
    
    area_intersection = np.bincount(intersection, minlength=K)
    area_output = np.bincount(output, minlength=K)
    area_target = np.bincount(target, minlength=K)
    area_union = area_output + area_target - area_intersection
    
    return area_intersection, area_union, area_target

def draw_bbox(image, cropped_image, bbox, gt_bbox, text, 
              cropped_mask, gt_mask, restored_mask,
              output_path, image_name, data_type,
              hr_box = None, restored_box=None,
              color=(0, 0, 255), text_color=(0, 0, 255), 
              thickness=4, font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1, font_thickness=2):
    """
    在原始图像上绘制边界框 (bbox) 并保存。

    :param image_path: 原始图像路径
    :param bboxes: 边界框列表，格式 [(x_min, y_min, x_max, y_max), ...]
    :param output_path: 结果图像的保存路径
    :param color: bbox 颜色 (B, G, R)，默认绿色
    :param thickness: bbox 线条粗细
    """
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_name = image_name.split(".")[0]
    output_path = os.path.join(output_path, image_name)
    output_path = os.path.join(output_path, data_type)
    os.makedirs(output_path, exist_ok=True)
    if restored_mask is not None: # tensor
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[:, :] = (0, 0, 255)  # 设定颜色 (默认红色)
        mask_indices = restored_mask > 0
        image[mask_indices] = cv2.addWeighted(image, 1 - 0.5, color_mask, 0.5, 0)[mask_indices]
        # 保存带框+掩码图像
        file_path = os.path.join(output_path, "visual_only_mask.jpg")
        cv2.imwrite(file_path, image)
        logger.info(f"裁剪图像+掩码已保存: {file_path}")
    # for bbox in bboxes:
    x_min, y_min, x_max, y_max = map(int, bbox)  # 确保坐标为整数
    
    # 计算文本尺寸
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    # 计算文本绘制位置：默认在bbox上方，且居中于bbox水平中点
    text_x = x_min + (x_max - x_min - text_width) // 2
    text_y = y_min - 10  # 在bbox上方留10个像素的间隔

    # 如果文本位置超出图像顶部，则将文本放到bbox内上方位置
    if text_y - text_height - baseline < 0:
        text_y = y_min + text_height + baseline + 10

    # 绘制文本背景（可选），增强文本可读性
    # 先绘制一个填充的矩形作为背景，再写文字
    bg_x1 = text_x
    bg_y1 = text_y - text_height - baseline
    bg_x2 = text_x + text_width
    bg_y2 = text_y + baseline
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)  # 白色背景

    # 绘制文本 
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness) # 送进入第二阶段进行裁剪的box
    
    if restored_box is not None: # 添加restored box 可视化结果
        restored_x_min, restored_y_min, restored_x_max, restored_y_max = map(int, restored_box)
        cv2.rectangle(image, (restored_x_min, restored_y_min), (restored_x_max, restored_y_max), color, thickness)
        
    if hr_box is not None: # 添加原始hr box 的可视化结果
        hr_x_min, hr_y_min, hr_x_max, hr_y_max = map(int, hr_box)
        cv2.rectangle(image, (hr_x_min, hr_y_min), (hr_x_max, hr_y_max), (255, 0, 0), thickness) # 蓝色 第一阶段直接出的box
    ### ADDED: 绘制 gt_bbox（绿色框）
    if gt_bbox is not None:
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = map(int, gt_bbox)
        cv2.rectangle(image, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0), thickness)  # 绿色框
        cv2.putText(image, "GT", (gt_xmin, gt_ymin - 10), font, font_scale, (0, 255, 0), font_thickness)
    # 保存带框图像
    
    
    file_path = os.path.join(output_path, "ori_imag_with_crop_box.jpg")
    cv2.imwrite(file_path, image)
    logger.info(f"带框图像已保存至: {file_path}")
    
    
    
    if gt_mask is not None: # tensor
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[:, :] = (0, 255, 0)  # 设定颜色 (默认绿色)
        mask_indices = gt_mask > 0
        image[mask_indices] = cv2.addWeighted(image, 1 - 0.5, color_mask, 0.5, 0)[mask_indices]
        # 保存带框+掩码图像
        file_path = os.path.join(output_path, "ori_imag_with_gt_mask.jpg")
        cv2.imwrite(file_path, image)
        logger.info(f"带框+gt掩码已保存: {file_path}")
    
    if cropped_image is not None and cropped_mask is not None: # tensor
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        color_mask = np.zeros_like(cropped_image, dtype=np.uint8)
        color_mask[:, :] = (0, 0, 255)  # 设定颜色 (默认红色)
        mask_indices = cropped_mask > 0
        cropped_image[mask_indices] = cv2.addWeighted(cropped_image, 1 - 0.5, color_mask, 0.5, 0)[mask_indices]
        # 保存裁剪图像
        file_path = os.path.join(output_path, "crop_img_with_pred_mask.jpg")
        cv2.imwrite(file_path, cropped_image)
        logger.info(f"裁剪图像+掩码已保存: {file_path}")

    if restored_mask is not None: # tensor
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[:, :] = (0, 0, 255)  # 设定颜色 (默认红色)
        mask_indices = restored_mask > 0
        image[mask_indices] = cv2.addWeighted(image, 1 - 0.5, color_mask, 0.5, 0)[mask_indices]
        # 保存带框+掩码图像
        file_path = os.path.join(output_path, "ori_imag_with_restored_mask.jpg")
        cv2.imwrite(file_path, image)
        logger.info(f"裁剪图像+掩码已保存: {file_path}")

def dynamic_box(bbox, image, min_size=512):
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    side = max(min_size, max(w, h))

    x_min_new = int(round(cx - side / 2))
    x_max_new = int(round(cx + side / 2))
    y_min_new = int(round(cy - side / 2))
    y_max_new = int(round(cy + side / 2))
    
    # restrict the box, not full the image size
    img_h, img_w = image.shape[:2]
    # if image_size is not None:
    # img_h, img_w = image_size
    x_min_new = max(0, min(x_min_new, img_w - 1))
    x_max_new = max(0, min(x_max_new, img_w - 1))
    y_min_new = max(0, min(y_min_new, img_h - 1))
    y_max_new = max(0, min(y_max_new, img_h - 1))
    
    return (x_min_new, y_min_new, x_max_new, y_max_new)

def restore_box(hr_bbox, lr_bbox, hr_image, lr_image):
    hr_h, hr_w = hr_image.shape[:2]
    lr_h, lr_w = lr_image.shape[:2]
    
    x_factor = lr_w/hr_w
    y_factor = lr_h/hr_h
    
    hr_x_min, hr_y_min, hr_x_max, hr_y_max = hr_bbox
    lr_x_min, lr_y_min, lr_x_max, lr_y_max = lr_bbox
    
    restore_x_min = lr_x_min + hr_x_min
    restore_y_min = lr_y_min + hr_y_min
    restore_x_max = lr_x_max + hr_x_min
    restore_y_max = lr_y_max + hr_y_min
    
    return (restore_x_min, restore_y_min, restore_x_max, restore_y_max)

def reform_messgae_HR(is_final_answer, input_question, question_type, options, image, resize):
    if is_final_answer:
        if question_type == "referring":
            QUESTION_TEMPLATE = \
                "Based on the '{Question}', identify a 256*256 bounding box that best localizes the region most relevant to the query. And respond with whether the object is found." \
                "Compare the difference between regions and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the 256*256 region bbox and the final response inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
                
            message = [{
                    "role": "user",
                    "content": [
                    {
                            "type": "image", 
                            "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                        },
                        {   
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=input_question, 
                            Answer="{'bbox': [x_min,y_min,x_min+256,y_min+256], 'response': 'The object is found.'}") # just a example not the real answer
                        }
                    ]
                }]
        elif question_type == "reasoning":
            QUESTION_TEMPLATE = \
                "Based on the '{Question}', identify a 256*256 bounding box that best localizes the region most relevant to the query. And give me a final response with a word or phrase." \
                "Compare the difference between regions and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the 256*256 region bbox and the final response inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
            
            message= [{
                    "role": "user",
                    "content": [
                    {
                            "type": "image", 
                            "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                        },
                        {   
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=input_question, 
                                    Answer="{'bbox': [x_min,y_min,x_min+256,y_min+256], 'response': 'The object is found.'}") # just a example not the real answer
                        }
                    ]
                }]
        elif question_type == "option":
            QUESTION_TEMPLATE = \
                "Based on the '{Question}', identify a 256*256 bounding box that best localizes the region most relevant to the query. And give me a correct option from {Options}." \
                "Compare the difference between regions and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the 256*256 region bbox and the final response inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
            
            message = [{
                    "role": "user",
                    "content": [
                    {
                            "type": "image", 
                            "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                        },
                        {   
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=input_question, 
                                    Answer="{'bbox': [x_min,y_min,x_min+256,y_min+256], 'response': 'A'}",
                                                                    Options=options) # just a example not the real answer
                        }
                    ]
                }]
        else:
            assert 1 == 2,  f"{question_type} error"
            
        
    else:
        QUESTION_TEMPLATE = \
            "Please find '{Question}' with bbox and points." \
            "Compare the difference between objects and find the most closely matched one." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"
        message = [{
            "role": "user",
            "content": [
            {
                    "type": "image", 
                    "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=input_question, 
                        Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}") # just a example not the real answer
                }
            ]
        }]
        
    return message

def reform_messgae_LR(is_final_answer, input_question, question_type, options, image, resize):
    if is_final_answer:
        if question_type == "referring":
            QUESTION_TEMPLATE = \
                "Please find '{Question}' with bbox and points. And respond with whether the object is found." \
                "Compare the difference between objects and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
                
            message = [{
                    "role": "user",
                    "content": [
                    {
                            "type": "image", 
                            "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                        },
                        {   
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=input_question, 
                            Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180], response: 'The object is found.'}") # just a example not the real answer
                        }
                    ]
                }]
        elif question_type == "reasoning":
            QUESTION_TEMPLATE = \
                "Please find '{Question}' with bbox and points. And give me a final response with a word or phrase." \
                "Compare the difference between objects and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
            
            message= [{
                    "role": "user",
                    "content": [
                    {
                            "type": "image", 
                            "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                        },
                        {   
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=input_question, 
                                    Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180], response: 'The object is white.' }") # just a example not the real answer
                        }
                    ]
                }]
        elif question_type == "option":
            QUESTION_TEMPLATE = \
            "Please find '{Question}' with bbox and points. And give me a correct option from {Options}." \
            "Compare the difference between objects and find the most closely matched one." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"
            
            message = [{
                    "role": "user",
                    "content": [
                    {
                            "type": "image", 
                            "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                        },
                        {   
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=input_question, 
                                    Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180], response: 'A'}",
                                    Options=options) # just a example not the real answer
                        }
                    ]
                }]
        else:
            assert 1 == 2,  f"{question_type} error"
            
        
    else:
        QUESTION_TEMPLATE = \
            "Please find '{Question}' with bbox and points." \
            "Compare the difference between objects and find the most closely matched one." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"
        message = [{
            "role": "user",
            "content": [
            {
                    "type": "image", 
                    "image": image.resize((resize[0], resize[1]), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=input_question, 
                        Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}") # just a example not the real answer
                }
            ]
        }]
        
    return message

def main():
    args = parse_args()
    
    logger.info("Loading Qwen model !!")
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    logger.info("Loading Cascade Qwen model !!")
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    cascade_reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.cascade_reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    logger.info("Loading SAM2 model !!")    
    segmentation_model = SAM2ImagePredictor(build_sam2(args.segmentation_config_path, args.segmentation_model_path))
    
    reasoning_model.eval()
    
    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    logger.info("Loading Annotations !!")
    annotations = load_data_from_json(args.test_json_path)    
    
    logger.info("Q-A Stage:{}".format(args.qa_stage))
    
    option_correct = 0
    option_num = 0
    reasoning_correct = 0
    reasoning_num = 0
    intersection_meter = []
    union_meter = []
    acc_iou_meter = []
    
    ######################## add full metric 
    seg_metric_dict = {"s":{"intersection_meter":[], "union_meter":[], "acc_iou_meter":[]}, # "giou":0, "ciou":0
                       "xs":{"intersection_meter":[], "union_meter":[], "acc_iou_meter":[]}, 
                       "xxs":{"intersection_meter":[], "union_meter":[], "acc_iou_meter":[]},
                       "all":{"intersection_meter":[], "union_meter":[], "acc_iou_meter":[]}}
    qa_metric_dict = {"option":{
                            "colors":{"num":0, "correct":0}, # "acc":0
                            "shape":{"num":0, "correct":0},
                            "others":{"num":0, "correct":0},
                            "position":{"num":0, "correct":0},
                            "avg":{"num":0, "correct":0}},
                      "reasoning":{
                            "colors":{"num":0, "correct":0},
                            "shape":{"num":0, "correct":0},
                            "others":{"num":0, "correct":0},
                            "position":{"num":0, "correct":0},
                            "avg":{"num":0, "correct":0}}}

    def update_seg_metric(seg_metric_dict, areas, intersection, union, acc_iou):
        # intersection_meter.append(intersection)
        # union_meter.append(union)
        # acc_iou_meter.append(acc_iou)
        XXS_TH = 0.017
        S_TH = 0.055
        if areas > S_TH:
            seg_metric_dict["s"]["intersection_meter"].append(intersection)
            seg_metric_dict["s"]["union_meter"].append(union)
            seg_metric_dict["s"]["acc_iou_meter"].append(acc_iou)
        elif areas < XXS_TH:
            seg_metric_dict["xxs"]["intersection_meter"].append(intersection)
            seg_metric_dict["xxs"]["union_meter"].append(union)
            seg_metric_dict["xxs"]["acc_iou_meter"].append(acc_iou)
        else:
            seg_metric_dict["xs"]["intersection_meter"].append(intersection)
            seg_metric_dict["xs"]["union_meter"].append(union)
            seg_metric_dict["xs"]["acc_iou_meter"].append(acc_iou)
        
        seg_metric_dict["all"]["intersection_meter"].append(intersection)
        seg_metric_dict["all"]["union_meter"].append(union)
        seg_metric_dict["all"]["acc_iou_meter"].append(acc_iou)
            
        return seg_metric_dict

    def update_qa_metric_correct(qa_metric_dict, data_type, attribute):
        qa_metric_dict[f"{data_type}"][f"{attribute}"]["correct"] = qa_metric_dict[f"{data_type}"][f"{attribute}"]["correct"] + 1
        # qa_metric_dict[f"{data_type}"][f"{attribute}"]["num"] = qa_metric_dict[f"{data_type}"][f"{attribute}"]["num"] + 1
        qa_metric_dict[f"{data_type}"]["avg"]["correct"] = qa_metric_dict[f"{data_type}"]["avg"]["correct"] + 1
        return qa_metric_dict

    def update_qa_metric_num(qa_metric_dict, data_type, attribute):
        qa_metric_dict[f"{data_type}"][f"{attribute}"]["num"] = qa_metric_dict[f"{data_type}"][f"{attribute}"]["num"] + 1
        return qa_metric_dict
    
    def final_metric(qa_metric_dict, seg_metric_dict, reasoning_num, option_num):
        qa_metric_dict["option"]["avg"]["num"] = option_num
        qa_metric_dict["reasoning"]["avg"]["num"] = reasoning_num
        
        for data_type in qa_metric_dict.keys():
            for attribute in qa_metric_dict[f"{data_type}"].keys():
                qa_metric_dict[f"{data_type}"][f"{attribute}"]["acc"] = float(float(qa_metric_dict[f"{data_type}"][f"{attribute}"]["correct"]) / (qa_metric_dict[f"{data_type}"][f"{attribute}"]["num"] + 1e-10))
        
        for scale in seg_metric_dict.keys():
            iou_class = sum(seg_metric_dict[scale]["intersection_meter"]) / (sum(seg_metric_dict[scale]["union_meter"]) + 1e-10)
            acc_iou_meter_sum  = sum(seg_metric_dict[scale]["acc_iou_meter"])
            seg_metric_dict[scale]["giou"] = acc_iou_meter_sum / len(seg_metric_dict[scale]["acc_iou_meter"])
            seg_metric_dict[scale]["ciou"] = iou_class[1]
            
        return qa_metric_dict, seg_metric_dict
    
    def save_metric(qa_metric_dict, seg_metric_dict, save_path):
        collected_metric = {"seg":{
                                "s":{}, # "giou":0, "ciou":0
                                "xs":{}, 
                                "xxs":{},
                                "all":{}}}
        
        collected_metric["seg"]["s"]["giou"] = float(seg_metric_dict["s"]["giou"][1])
        collected_metric["seg"]["xs"]["giou"] = float(seg_metric_dict["xs"]["giou"][1])
        collected_metric["seg"]["xxs"]["giou"] = float(seg_metric_dict["xxs"]["giou"][1])
        collected_metric["seg"]["all"]["giou"] = float(seg_metric_dict["all"]["giou"][1])
        
        collected_metric["seg"]["s"]["ciou"] = float(seg_metric_dict["s"]["ciou"])
        collected_metric["seg"]["xs"]["ciou"] = float(seg_metric_dict["xs"]["ciou"])
        collected_metric["seg"]["xxs"]["ciou"] = float(seg_metric_dict["xxs"]["ciou"])
        collected_metric["seg"]["all"]["ciou"] = float(seg_metric_dict["all"]["ciou"])
        
        collected_metric["qa"] = qa_metric_dict
        
        with open(os.path.join(save_path, "results_all.json"), "w") as f:
            pass
        
        with open(os.path.join(save_path, "results_all.json"), "w") as f:
            json.dump(collected_metric, f,  indent=4)
        
        logger.info(collected_metric)
    ########################################################################
    
    iter_save = 0
    box_is_valid_num = 0
    all_num = 0
    # annotations = annotations[30:34]
    # annotations = annotations[126:129]
    for anno in tqdm.tqdm(annotations):
        # if anno["Q-type"] != "option":
        #     continue
        question_type  = anno["Q-type"]
        input_question = anno["Q"]
        gt_answer   = anno["A"]
        mask_points = anno["points"]
        image_name = anno["image_path"]
        ################
        if "options" in anno.keys():
            options = anno["options"]
        else:
            options = None
        attribute = anno["attribute"]
        # if attribute != "others" or question_type != "reasoning":
        #     continue
        # else:
        #     iter_save = iter_save + 1
        area_percent = anno["area_percent"]
        ################
        image_path = os.path.join(args.image_path, image_name)
        
        logger.info("Reading image from: {}".format(image_path))    
        
        image_np = cv2.imread(image_path)
        image_ori = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        mask_json = get_mask_from_points(anno, image_np)

        logger.info("User question: {}".format(input_question))
        if options != None:
            logger.info("options: {}".format(input_question))
        
        # image = PILImage.open(image_path)
        image = Image.fromarray(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
        original_width, original_height = image.size
        
        ######## prepare for image size ########
        resize_size = ast.literal_eval(args.resize_size) # w h
        logger.info("resize into :{}".format(resize_size))
        ################################################
        
        x_factor, y_factor = original_width/resize_size[0], original_height/resize_size[1]
        
        message = reform_messgae_HR(True, input_question, question_type, options, image, resize_size)

        # logger.info("stage 1 messages:{}".format(message))
        messages = []
        messages.append(message)

        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        
        #pdb.set_trace()
        image_inputs, video_inputs = process_vision_info(messages)
        #pdb.set_trace()
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # print('inputs',inputs)
        start_time = time.time()
        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        end_time = time.time()
        print(f"stage1 耗时: {end_time - start_time:.2f} 秒")
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        logger.info("Model output:{}".format(output_text))
        
        if args.qa_stage == "stage1":
            bbox, points, think, final_response = extract_bbox_points_think(output_text[0], x_factor, y_factor, True)
            logger.info("hr bbox:{}".format(bbox))
        elif args.qa_stage == "stage2" or args.qa_stage == "no_qa":
            bbox, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
                
        ####################  cascade reasoning process ####################
        
        #################### change box size and location based on first output 
        cascade_resize_size = ast.literal_eval(args.cascade_resize_size) # w h 需要确认一个点在于级联模型所使用的size是多少
        if bbox != None:
            hr_bbox = bbox.copy()
            if args.dynamic_box:
                bbox = dynamic_box(bbox, image_ori, cascade_resize_size[0]) # 根据box情况进行调整
            ####################
            
            x_min, y_min, x_max, y_max = bbox
            cropped_image = image_ori[int(y_min):int(y_max), int(x_min):int(x_max)]  # 裁剪图像
            
            logger.info("cascade resize into :{}".format(cascade_resize_size))
            logger.info("cropped_image size :{}".format(cropped_image.shape))
            try:
                image_for_cascade = Image.fromarray(cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB))
            except:
                logger.info("box is error :{}".format(cropped_image.shape))
                image_for_cascade = Image.fromarray(cv2.cvtColor(image_ori,cv2.COLOR_BGR2RGB))
                cropped_image = image_ori
                width, height = image_for_cascade.size
                x_min, y_min, x_max, y_max = 0, 0, width, height
            cascade_original_width, cascade_original_height = image_for_cascade.size
            cascade_x_factor, cascade_y_factor = cascade_original_width/cascade_resize_size[0], cascade_original_height/cascade_resize_size[1]
            
            cascade_messages = []
            

            cascade_message = reform_messgae_LR(True, input_question, question_type, options, image_for_cascade, cascade_resize_size) # 手动改 yjz

                
            cascade_messages.append(cascade_message)

            # Preparation for inference
            cascade_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in cascade_messages]
            
            #pdb.set_trace()
            cascade_image_inputs, cascade_video_inputs = process_vision_info(cascade_messages)
            #pdb.set_trace()
            cascade_inputs = processor(
                text=cascade_text,
                images=cascade_image_inputs,
                videos=cascade_video_inputs,
                padding=True,
                return_tensors="pt",
            )
            cascade_inputs = cascade_inputs.to("cuda")
            start_time = time.time()
            # Inference: Generation of the output
            cascade_generated_ids = cascade_reasoning_model.generate(**cascade_inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            end_time = time.time()
            print(f"stage2 耗时: {end_time - start_time:.2f} 秒")
            cascade_generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(cascade_inputs.input_ids, cascade_generated_ids)
            ]
            cascade_output_text = processor.batch_decode(
                cascade_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            logger.info("Cascade model output:{}".format(cascade_output_text))
            
            # cascade_bbox, points, think, final_response = extract_bbox_points_think(cascade_output_text[0], cascade_x_factor, cascade_y_factor, is_final_answer_model=True)
            if args.qa_stage == "stage2":
                cascade_bbox, points, think, final_response = extract_bbox_points_think(cascade_output_text[0], cascade_x_factor, cascade_y_factor, True)
            elif args.qa_stage == "stage1" or args.qa_stage == "no_qa":
                try:
                    cascade_bbox, points, think = extract_bbox_points_think(cascade_output_text[0], cascade_x_factor, cascade_y_factor, False)
                except: 
                    cascade_bbox, points, think = None, None, ""
            if args.qa_stage != "no_qa":
                final_response = final_response.lower()
                gt_answer = gt_answer.lower()
                if question_type == "option":
                    logger.info("-option-")
                    logger.info('final_response: {}'.format(final_response))
                    logger.info('gt_answer: {}'.format(gt_answer))
                    # 匹配 b / (b) / [b] / 'b' / "b"，同时考虑标点、空格分隔，避免误命中其它词
                    pattern = rf"(?:\b|[\(\[\{{'\" ]){gt_answer}(?:\b|[\)\]\}}'\" ,.!?])"
                    if re.search(pattern, final_response):
                        option_correct = option_correct + 1 # update_qa
                        qa_metric_dict = update_qa_metric_correct(qa_metric_dict, question_type, attribute)
                        logger.info("选择正确")
                    qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute)
                    option_num = option_num + 1
                if question_type == "reasoning":
                    logger.info("-reasoning-")
                    logger.info('final_response: {}'.format(final_response))
                    logger.info('gt_answer: {}'.format(gt_answer))
                    final_response = final_response.split()
                    for word in final_response: # 这里可能进行多次循环
                        similarity = difflib.SequenceMatcher(None, word, gt_answer).ratio()
                        if similarity >= 0.8:
                            reasoning_correct = reasoning_correct + 1
                            qa_metric_dict = update_qa_metric_correct(qa_metric_dict, question_type, attribute)
                            logger.info("推理正确")
                            # logger.info('reasoning_correct111{}'.format(reasoning_correct))
                            break
                            
                    qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute) # 错误的地方也要更新
                    reasoning_num = reasoning_num + 1  # 更新总数
                    # logger.info('reasoning_num111{}'.format(reasoning_num))
            else:
                logger.info("No QA, No metric update!!")

            ##################################################################
            
            gt_bbox = get_bbox_from_mask(mask_json, 
                            original_width, 
                            original_height)
            
            ## 比较两个box bbox时生成box 
            if is_inside(gt_bbox, bbox):
                box_is_valid_num  = box_is_valid_num + 1 
            all_num = all_num + 1
            if cascade_bbox is None and points is None:
                logger.info("No bbox and points, create empty mask !!")
                mask = np.zeros((mask_json.shape[0], mask_json.shape[1])).astype(bool)
                restored_mask = mask.copy()
            else:
                logger.info("Thinking bbox:{}".format(cascade_bbox))
                cascade_x_min, cascade_y_min, cascade_x_max, cascade_y_max = cascade_bbox
                # cropped_image = image_ori[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                logger.info("Thinking process:{}".format(think))
                
                restore_bbox = restore_box(bbox, cascade_bbox, image_ori, cropped_image)
                start_time = time.time()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    segmentation_model.set_image(cropped_image) # set cropped image as input
                    masks, scores, _ = segmentation_model.predict(
                        point_coords=points,
                        point_labels=[1,1],
                        box=cascade_bbox
                    )
                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                end_time = time.time()
                print(f"sam2 耗时: {end_time - start_time:.2f} 秒")
                
                # logger.info("Masks:{}".format(masks.shape)) # numpy
                mask = masks[0].astype(bool)
                
                ####################### mask restore #######################
                # cascade_restored_mask = np.zeros((cropped_image.shape[0], cropped_image.shape[1]))
                # logger.info("cascade_restored_mask shape:{}".format(cascade_restored_mask.shape)) # should be same as cropped image size
                # cascade_restored_mask[int(cascade_y_min):int(cascade_y_max), int(cascade_x_min):int(cascade_x_max)] = mask
                # cascade_restored_mask = cascade_restored_mask.astype(bool)
                
                restored_mask = np.zeros((image_ori.shape[0], image_ori.shape[1]))
                logger.info("restored_mask shape:{}".format(restored_mask.shape)) # should be same as cropped image size
                restored_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = mask
                restored_mask = restored_mask.astype(bool)
                ###################################################################
                
            logger.info("mask:{}".format(mask.shape))    
            logger.info("mask_json: {}".format(mask_json.shape))    
            
            
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            intersection_i, union_i, _ = intersectionAndUnionCPU(
                restored_mask, mask_json, 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
            
            intersection_meter.append(intersection)
            union_meter.append(union)
            acc_iou_meter.append(acc_iou)
            
            seg_metric_dict = update_seg_metric(seg_metric_dict, 
                                                area_percent, 
                                                intersection,
                                                union,
                                                acc_iou)
            
            if args.save_results and cascade_bbox is not None and points is not None:
                draw_bbox(image_ori, cropped_image, bbox, gt_bbox, 
                        input_question, mask, mask_json, restored_mask,
                        args.save_path, image_name, question_type,
                        hr_bbox, restore_bbox)
            iter_save = iter_save + 1
            # print('==================',iter_save)
        else:  # 第一阶段box没预测出来
            cascade_bbox = None
            logger.info("No bbox and points, create empty mask !!")
            mask = np.zeros((mask_json.shape[0], mask_json.shape[1])).astype(bool)
            restored_mask = mask.copy()
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            intersection_i, union_i, _ = intersectionAndUnionCPU(
                restored_mask, mask_json, 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
            
            intersection_meter.append(intersection)
            union_meter.append(union)
            acc_iou_meter.append(acc_iou)
            
            seg_metric_dict = update_seg_metric(seg_metric_dict, 
                                                area_percent, 
                                                intersection,
                                                union,
                                                acc_iou)
            all_num = all_num + 1
            if question_type == "option":
                option_num = option_num + 1
                qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute)
            if question_type == "reasoning":
                reasoning_num = reasoning_num + 1
                qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute) # 
            # logger.info('reasoning_num2{}'.format(reasoning_num))
    # print(intersection_meter, union_meter, acc_iou_meter)
    iou_class = sum(intersection_meter) / (sum(union_meter) + 1e-10)
    ciou = iou_class[1]
    acc_iou_meter_sum = sum(acc_iou_meter)
    giou = acc_iou_meter_sum / len(acc_iou_meter)
    box_valid_num_acc = box_is_valid_num / all_num
    # logger.info("iter_save:{}".format(iter_save))
    if option_num==0:
        print("no options")
        option_acc = -1
    else:
        option_acc = option_correct / option_num
        
    if reasoning_num==0:
        print("no reasoning_num")
        reasoning_acc = -1
    else: 
        reasoning_acc = reasoning_correct / reasoning_num
    
    logger.info("intersection_meter.sum:{}".format(sum(intersection_meter)))
    logger.info("union_meter.sum:{}".format(sum(union_meter)))
    logger.info("acc_iou_meter.avg:{}".format(acc_iou_meter_sum / len(acc_iou_meter)))
    
    logger.info("giou: {}, ciou: {}".format(giou, iou_class))
    logger.info("giou: {:.4f}, ciou: {:.4f}".format(giou[1], ciou))
    logger.info("box_valid_num_acc: {:.4f}, option_acc: {:.4f}, reasoning_acc: {:.4f}".format(box_valid_num_acc, option_acc, reasoning_acc))
    
    qa_metric_dict, seg_metric_dict = final_metric(qa_metric_dict, seg_metric_dict, reasoning_num, option_num)
    # logger.info("seg_metric:{}, qa_metric:{}".format(seg_metric_dict, qa_metric_dict))
    
    if args.save_results:
        save_metric(qa_metric_dict, seg_metric_dict, args.save_path)
        result = {"giou":float(giou[1]), "ciou":float(ciou), "box_valid_num_acc": float(box_valid_num_acc), 
                  "option_acc": float(option_acc), "reasoning_acc": float(reasoning_acc)}
        # os.makedirs(args.save_path)
        with open(os.path.join(args.save_path, "results.json"), "w") as f:
            pass
        
        with open(os.path.join(args.save_path, "results.json"), "w") as f:
            json.dump(result, f,  indent=4)
        
    

if __name__ == "__main__":
    main()
