import json
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from PIL import Image
import io
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()

def draw_bbox(image, cropped_image, bbox, text, 
              cropped_mask, gt_mask, restored_mask,
              output_path, image_name, data_type,
              color=(0, 0, 255), text_color=(0, 0, 255), 
              thickness=4, font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1, font_thickness=2):

    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # 保存带框图像
    image_name = image_name.split(".")[0]
    output_path = os.path.join(output_path, image_name)
    output_path = os.path.join(output_path, data_type)
    os.makedirs(output_path, exist_ok=True)
    
    file_path = os.path.join(output_path, "ori_imag_with_crop_box.jpg")
    cv2.imwrite(file_path, image)
    logger.info(f"带框图像已保存至: {file_path}")
    
        
def load_image_as_bytes(path, image_name, new_size):
    # path = "/13390024681/yjz/All_Data_HD_Reasoning/MiniSeg_ft2_raw/all_images"
    image_path = os.path.join(path, image_name)
    if os.path.exists(image_path) and os.path.isfile(image_path):
        try:
            # with Image.open(image_path) as img:
            image_np = cv2.imread(image_path)
            img = Image.fromarray(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
            img = img.resize(new_size)
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    else:
        print("Path not exist:{}".format(image_path))
        return None

def get_solution_from_box_ref_seg(box, res, best_region):
    x_min, y_min, x_max, y_max = map(int, box)
    x_min_pre, y_min_pre, x_max_pre, y_max_pre = map(int, best_region)
    # 计算中心点
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # 计算允许的最大偏移范围（box大小的1/10）
    dx = (x_max - x_min) // 10
    dy = (y_max - y_min) // 10

    # 生成随机偏移，但确保偏移后仍在 box 内
    px = min(max(cx + random.randint(-dx, dx), x_min), x_max)
    py = min(max(cy + random.randint(-dy, dy), y_min), y_max)
    
    # forms = "<box>({x_min},{y_min}),({x_max},{y_max})</box><points>({cx},{cy}),({px},{py})</points>"
    forms = "<box>({x_min},{y_min}),({x_max},{y_max})</box><points>({cx},{cy}),({px},{py})</points><response>{res}</response><best_region>({x_min_pre},{y_min_pre}),({x_max_pre},{y_max_pre})</best_region>"

    
    return forms.format(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, 
                        cx=cx, cy=cy, px=px, py=py, res=res, x_min_pre=x_min_pre, y_min_pre=y_min_pre, x_max_pre=x_max_pre, y_max_pre=y_max_pre)

def resize_bbox(bbox, original_size, new_size):
    """
    按照图像 resize 比例缩放 bbox。
    
    :param bbox: (x_min, y_min, x_max, y_max) 原始 bbox 坐标
    :param original_size: (height, width) 原始图像尺寸
    :param new_size: (new_height, new_width) 目标 resize 尺寸
    :return: (x_min', y_min', x_max', y_max') 变换后的 bbox
    """
    orig_h, orig_w = original_size
    new_h, new_w = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    scale = min(scale_y,scale_x)
    nw = int(orig_w * scale)
    nh = int(orig_h * scale)
    dx = (new_w - nw) // 2
    dy = (new_h - nh) // 2
    
    x_min, y_min, x_max, y_max = bbox
    x_min = int(x_min * scale_x + dx)
    y_min = int(y_min * scale_y + dy)
    x_max = int(x_max * scale_x + dx)
    y_max = int(y_max * scale_y + dy)

    return (x_min, y_min, x_max, y_max)

def get_mask_from_points(points, image_path):
    
    # img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    img = cv2.imread(image_path)
    # img = Image.fromarray(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
    height, width = img.shape[:2]
    label_value = 1  # target

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
    cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)
    
    return mask, height, width

def get_bbox_from_mask(points, image, new_size, resize_box=False):
    """
    通过二值 mask 计算外接矩形框 (bbox)。
    :param mask: 二值化 mask (numpy array)
    :return: (x_min, y_min, x_max, y_max) - 外接矩形框坐标
    """
    mask, height, width = get_mask_from_points(points, image)
    # logger.info("ori size:{}, new size:{}".format((height, width), new_size))
    
    # 找到 mask 中所有非零点的坐标
    y_coords, x_coords = np.nonzero(mask)  
    
    x_min = x_coords.min()  
    x_max = x_coords.max()  
    y_min = y_coords.min()  
    y_max = y_coords.max()
    
    # Calculate scale factors
    scale_x = new_size[1] / width
    scale_y = new_size[0] / height
    
    # Adjust the bounding box according to the scaling
    new_x_min = x_min * scale_x
    new_y_min = y_min * scale_y
    new_x_max = x_max * scale_x
    new_y_max = y_max * scale_y
    
    return (new_x_min, new_y_min, new_x_max, new_y_max)

def resize_bbox_with_factor(image_path, best_region, new_size):
    
    # img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    img = cv2.imread(image_path)
    # img = Image.fromarray(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
    height, width = img.shape[:2]
    
 
    
    x_min = best_region[0]
    x_max = best_region[2]
    y_min = best_region[1]
    y_max = best_region[3]
    
    # Calculate scale factors
    scale_x = new_size[1] / width
    scale_y = new_size[0] / height
    
    # Adjust the bounding box according to the scaling
    new_x_min = x_min * scale_x
    new_y_min = y_min * scale_y
    new_x_max = x_max * scale_x
    new_y_max = y_max * scale_y
    
    return (new_x_min, new_y_min, new_x_max, new_y_max)

def get_id_from_name(name):
    id = name.split(".")[0]
    return id


def save_in_chunks(dataset, chunk_size, save_path):
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset.select(range(i, min(i + chunk_size, len(dataset))))
        chunk.to_parquet(f"{save_path}/{i // chunk_size}.parquet")
        # table = pa.Table.from_pandas(chunk.to_pandas())
        # pq.write_table(table, f"{output_prefix}_part_{i // chunk_size}.parquet")
        
def json2parquet(json_path, save_path, split, num_limit, all_image_path, new_size, chunk_size, resize_box=False):
    logger.info("resize image to:{}".format(new_size))
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    anno = data["annotations"]
    # anno = anno[:40]
    train_data =[]
    count = 0
    for ann in tqdm(anno):
        train_ann = {}
        id = get_id_from_name(ann["image_path"])
        problem = ann["Q"]
        answer = ann["A"]
        points = ann["points"]
        type = ann["Q-type"]
        best_region = ann["best_region"]  # 需要处理size
        image_data = load_image_as_bytes(all_image_path, ann["image_path"], new_size) # image open format
        bbox = get_bbox_from_mask(ann["points"], 
                                  os.path.join(all_image_path, ann["image_path"]), 
                                  (new_size[1], new_size[0]),
                                  resize_box)
        logger.info("bbox {} ".format(bbox))
        logger.info("best_region {} ".format(best_region))
        resized_best_region = resize_bbox_with_factor(os.path.join(all_image_path, ann["image_path"]), best_region, (new_size[1], new_size[0]))
        logger.info("resized_best_region {} ".format(resized_best_region))
        # solution = get_solution_from_box(bbox)
        if type == "option":
            res = "{}".format(ann["A"])
            options = ann["options"]
        elif type == "referring":
            res = "The object is found."
        else:
            res = "{}".format(ann["A"])
        assert res, "res should not be empty!"
            # print(res)
        solution = get_solution_from_box_ref_seg(bbox, res, resized_best_region)
        if type == "option":
            train_ann.update({"id":id, "problem":problem, "solution":solution, "answer":answer, "type":type, "image":image_data, "options": options})
        else:
            train_ann.update({"id":id, "problem":problem, "solution":solution, "answer":answer, "type":type, "image":image_data, "options": None})
        logger.info("solution {} ".format(solution))
        # train_ann.update({"id":id, "problem":problem, "solution":solution, "answer":answer,"points":points, "type":type, "image":image_data})
        train_data.append(train_ann)
        count = count + 1
        if num_limit is not None:
            if count > num_limit:
                break
        
    ds = Dataset.from_list(train_data)
    logger.info("Saved into {} chunks".format(chunk_size))
    # ds.to_parquet(f"{save_path}/{split}.parquet") # split决定是分割训练集还是测试集
    for i in range(0, len(ds), chunk_size):
        chunk = ds.select(range(i, min(i + chunk_size, len(ds))))
        chunk.to_parquet(f"{save_path}/{split}_{i // chunk_size}.parquet")

    print(f"JSON 数据和图片已成功转换为 {save_path}")

def test_load(data_path):
    
    data = load_dataset(data_path)["train"]
    print(data)
    
    count = 0 
    for anno in data:
        print(anno.keys())
        print(anno["image"]) # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=840x840 at 0x7F7786428650> size需要保持
        print(anno["id"])
        print(anno["problem"])
        print(anno["solution"])
        # print(anno["img_height"])
        # print(anno["img_width"])
        count = count + 1
        if count > 20 :
            break

if __name__ == "__main__":

    args = parse_args()
    all_image_path = "/15324359926/seg/HD_Reasoning-main-yjz/All_final_data/all_images"

    
    data_path = "/15324359926/seg/HD_Reasoning-main-yjz/All_final_data/all_json/training_v4_with_best_region_with_qa/training_v4_with_best_region_with_qa_merge.json"
    save_path = "/15324359926/seg/HD_Reasoning-main-yjz/All_final_data/all_parquet_with_best_region_by_LR_model/train_v4_defalut_random512region_best_region"
    
    # data_path = "/13390024681/yjz/All_Data_HD_Reasoning/MiniSeg/wrong_set.json"
    # save_path = "/13390024681/yjz/All_Data_HD_Reasoning/MiniSeg/wrong_set"
    
    json2parquet(data_path, save_path, "train", None, all_image_path, (1920, 1080), 20, True)
    
    # test_load("/13390024681/yjz/All_Data_HD_Reasoning/MiniSeg/wrong_set") # 保存后即可通过文件夹路径访问
    # test_load("/13390024681/yjz/refCOCOg_2k_840")