import re
import json
import math
import pdb
import difflib
def seg_thinking_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def seg_segmentation_format_reward(predict_str: str) -> float:
    def is_valid_format(predict_str: str) -> bool:
        try:
            json_match = re.search(r'{[^}]+}', predict_str)
            if not json_match:
                return False
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # check the required keys
            required_keys = ['bbox', 'response']
            for key in required_keys:
                if key not in data:
                    return False
            
            # check the format of the value
            bbox = data['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False
                
            # points_1 = data['points_1']
            # points_2 = data['points_2']
            # if not isinstance(points_1, list) or len(points_1) != 2:
            #     return False
            # if not isinstance(points_2, list) or len(points_2) != 2:
            #     return False

            # 提取 response 部分
            response = data['response']
            # print('222response',response)

            if not isinstance(response, str) or response.strip() == '':
                # print('222response',response)
                return False
            print("-------------------------format is true ---------------------")
            return True
        except Exception:
            return False
    return 1.0 if is_valid_format(predict_str) else 0.0

def seg_iou_reward(predict_str: str, ground_truth: str) -> float:
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
        area2 = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
        union = area1 + area2 - inter
        return float(inter)/union
    
    try:
        ground_truth = ground_truth.strip()
        # gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        # gt_match = re.search(gt_box_pattern, ground_truth)
        gt_region_pattern = r'<best_region>\((\d+),(\d+)\),\((\d+),(\d+)\)</best_region>'
        gt_match_region = re.search(gt_region_pattern, ground_truth)
        if gt_match_region:
            gt_region = [int(gt_match_region.group(1)), int(gt_match_region.group(2)), int(gt_match_region.group(3)), int(gt_match_region.group(4))]
            print('--------------------------------gt_region-------------------------------','gt_region{}'.format(gt_region))
        # if gt_match:
        #     gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]

        
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        # pdb.set_trace()
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                if iou(content_bbox, gt_region) > 0.9: #change
                    print('--------------------------------box iou 1.0-------------------------------','content_bbox{}'.format(content_bbox),'gt_region{}'.format(gt_region))
                    return 1.0
    except Exception:
        pass
    return 0.0


# box in box and resize 
def seg_box_in_box_and_size_reward(predict_str: str, ground_truth: str) -> float:
    def is_valid_bbox(content_bbox, gt_bbox):
        # 真实框必须完全在预测框内
        if not (content_bbox[0] <= gt_bbox[0] and content_bbox[1] <= gt_bbox[1] and 
                content_bbox[2] >= gt_bbox[2] and content_bbox[3] >= gt_bbox[3]):
            return False

        # 计算预测框的宽高
        pred_width = content_bbox[2] - content_bbox[0] + 1
        pred_height = content_bbox[3] - content_bbox[1] + 1
        gt_width = gt_bbox[2] - gt_bbox[0] + 1
        gt_height = gt_bbox[3] - gt_bbox[1] + 1

        # 预测框大小限制
        # max_width, max_height = 1920 // 4, 1080 // 4  # (480, 270)
        if pred_width > 260 or pred_height > 260:
            return False
        if pred_width < gt_width * 2 or pred_height < gt_height * 2:
            return False

        # 预测框不能超出图像边界
        if not (0 <= content_bbox[0] < 1920 and 0 <= content_bbox[1] < 1080 and
                0 < content_bbox[2] <= 1920 and 0 < content_bbox[3] <= 1080):
            return False
        print('--------------------------------seg_box_in_box_and_size_reward--------------------------------','content_bbox{}'.format(content_bbox),'gt_bbox{}'.format(gt_bbox))
        return True

    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), 
                       int(gt_match.group(3)), int(gt_match.group(4))]
        else:
            return 0.0  # 解析失败则不给奖励
        
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            if 'bbox' in data and len(data['bbox']) == 4:
                content_bbox = data['bbox']
                if is_valid_bbox(content_bbox, gt_bbox):
                    return 1.0
    except Exception:
        pass
    return 0.0

# size_only 
def seg_box_size_only_reward(predict_str: str) -> float:
    def bbox_score(content_bbox):
        # 计算预测框的宽高
        pred_width = content_bbox[2] - content_bbox[0] + 1
        pred_height = content_bbox[3] - content_bbox[1] + 1

        # 目标尺寸限制（尽量靠近 256x256）
        target_size = 256
        tolerance = 15  # 可调节容忍误差  

        width_valid = target_size - tolerance <= pred_width <= target_size + tolerance
        height_valid = target_size - tolerance <= pred_height <= target_size + tolerance

        # 同时满足宽和高得1.0，任意满足一个得0.5，否则得0.0
        if width_valid and height_valid:
            return 1.0
        elif width_valid or height_valid:
            return 0.5
        else:
            return 0.0

    try:
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                score = bbox_score(content_bbox)
                print('-------------------seg_box_size_only_reward-------------------', 
                      'content_bbox:{}'.format(content_bbox) , 'score:{}'.format(score))
                return score
    except Exception:
        pass
    return 0.0

# box in box only 
def seg_box_in_box_only_reward(predict_str: str, ground_truth: str) -> float:
    def is_valid_bbox(content_bbox, gt_bbox):
        # 真实框必须完全在预测框内
        if not (content_bbox[0] <= gt_bbox[0] and content_bbox[1] <= gt_bbox[1] and 
                content_bbox[2] >= gt_bbox[2] and content_bbox[3] >= gt_bbox[3]):
            return False

        return True

    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), 
                       int(gt_match.group(3)), int(gt_match.group(4))]
            print('--------------------------------gt_bbox-------------------------------','gt_bbox{}'.format(gt_bbox))
        else:
            return 0.0  # 解析失败则不给奖励
        
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            if 'bbox' in data and len(data['bbox']) == 4:
                content_bbox = data['bbox']
                if is_valid_bbox(content_bbox, gt_bbox):
                    print('--------------------------------boxinbox-1.0-------------------------------','content_bbox{}'.format(content_bbox),'gt_bbox{}'.format(gt_bbox))

                    return 1.0
    except Exception:
        pass
    return 0.0

def get_region_with_gt_box(gt_box, image_width=1920, image_height=1080, region_size=256):
    x_min, y_min, x_max, y_max = gt_box

    # 计算GT box中心
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # 初步设定region范围（以中心点为正中心）
    left = int(center_x - region_size / 2)
    top = int(center_y - region_size / 2)
    right = left + region_size
    bottom = top + region_size

    # 如果超出左上边界，向右下平移
    if left < 0:
        right += -left
        left = 0
    if top < 0:
        bottom += -top
        top = 0

    # 如果超出右下边界，向左上平移
    if right > image_width:
        left -= (right - image_width)
        right = image_width
    if bottom > image_height:
        top -= (bottom - image_height)
        bottom = image_height

    # # 最后检查，保证GT box在区域内
    # assert left <= x_min and right >= x_max, f"GT box X轴超出区域: {left}-{right} vs {x_min}-{x_max}"
    # assert top <= y_min and bottom >= y_max, f"GT box Y轴超出区域: {top}-{bottom} vs {y_min}-{y_max}"

    return [left, top, right, bottom]

def seg_box_l1_reward(predict_str: str, ground_truth: str) -> float:
    # 已改成region
    # 扩展gt_box为region
    
    def l1_distance(box1, box2):
        return (abs(box1[0]-box2[0]) + abs(box1[1]-box2[1]) + abs(box1[2]-box2[2]) + abs(box1[3]-box2[3])) / 4
    
    try:
        ground_truth = ground_truth.strip()
        gt_region_pattern = r'<best_region>\((\d+),(\d+)\),\((\d+),(\d+)\)</best_region>'
        gt_match_region = re.search(gt_region_pattern, ground_truth)
        if gt_match_region:
            gt_region = [int(gt_match_region.group(1)), int(gt_match_region.group(2)), int(gt_match_region.group(3)), int(gt_match_region.group(4))]
            print('--------------------------------gt_region-------------------------------','gt_region{}'.format(gt_region))
            # assert  1  == 2
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                if l1_distance(content_bbox, gt_region) < 20:  #change
                    print('--------------------------------box l1 1.0-------------------------------','content_bbox{}'.format(content_bbox),'gt_region{}'.format(gt_region))
                    return 1.0
    except Exception:
        pass
    return 0.0

def seg_point_l1_reward(predict_str: str, ground_truth: str) -> float:
    def points_in_box(point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
    
    def points_distance(points1, points2):
        dist1 = math.sqrt((points1[0][0]-points2[0][0])**2 + (points1[0][1]-points2[0][1])**2) + \
                math.sqrt((points1[1][0]-points2[1][0])**2 + (points1[1][1]-points2[1][1])**2)
        
        dist2 = math.sqrt((points1[0][0]-points2[1][0])**2 + (points1[0][1]-points2[1][1])**2) + \
                math.sqrt((points1[1][0]-points2[0][0])**2 + (points1[1][1]-points2[0][1])**2)
        return min(dist1, dist2) / 2
        
    try: 
        gt_points_pattern = r'<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>'
        gt_match = re.search(gt_points_pattern, ground_truth)
        if gt_match:
            gt_points = [[int(gt_match.group(1)), int(gt_match.group(2))], [int(gt_match.group(3)), int(gt_match.group(4))]]
            
        json_pattern = r'{[^}]+}' 
        json_match = re.search(json_pattern, predict_str)

        if json_match:
            data = json.loads(json_match.group(0))
            # find bbox key
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
            # find points key
            points_keys = ['points_1', 'points_2']  # get the first two points keys
            if len(points_keys) == 2:
                point1 = data[points_keys[0]]
                point2 = data[points_keys[1]]
                point1 = [int(point1[0]), int(point1[1])]
                point2 = [int(point2[0]), int(point2[1])]
                if points_in_box(point1, content_bbox) and points_in_box(point2, content_bbox):
                    if points_distance([point1, point2], gt_points) < 100:
                        return 1.0
    except Exception:
        pass  # Continue to next verification method if this fails
    return 0.0

def extract_response(ground_truth: str) -> str:
    # 使用正则表达式提取 response 部分
    match = re.search(r"<response>(.*?)</response>", ground_truth)
    if match:
        return match.group(1).strip()  # 返回提取的 response 内容
    return ""

def compute_text_reward(predict_str: str, ground_truth: str) -> float: # json 必须双引号
    json_pattern = r'{[^}]+}' 
    json_match = re.search(json_pattern, predict_str)
    # print(json_match)
    try:
        if json_match:
            data = json.loads(json_match.group(0))
            output_text = data["response"]
            # print('output_text=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-',output_text)

            gt_response = extract_response(ground_truth).lower()
            assert gt_response, "gt_response should not be empty!"
            # print('gt_response=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-',gt_response)
            ## 返回三种gt_response 根据gt判断q_type(省事儿) 空-refferring 单词-qa 选项-option 后期优化传参数q_type方式
            # if q_type == "referring":
            if gt_response == "The object is found.".lower():
                # print("---referring---")
                referring_keywords = ["is found","is detected"] # 目前的数据全都能找到/ 后期可以加入no targets
                print('-------------------------------referring正确-------------------------------')

                return 1.0 if any(k in output_text.lower() for k in referring_keywords) else 0.0

            # elif q_type == "qa_op_seg":
            elif gt_response.lower() in ["a", "b", "c", "d", "A", "B", "C", "D"]:
                """
                    匹配多选题答案，gt_response 应为 "A", "B", "C" 等大写字母。
                    可匹配：b, (b), [b], 'b', "b" 等格式，同时避免误匹配 bark, abc 等单词。
                """
                output_text = output_text.lower()
                option = gt_response.lower()

                # 匹配 b / (b) / [b] / 'b' / "b"，同时考虑标点、空格分隔，避免误命中其它词                

                pattern = rf"(?:\b|[\(\[\{{'\" ]){option}(?:\b|[\)\]\}}'\" ,.!?])"
                if re.search(pattern, output_text):
                    print('-------------------------------option正确-------------------------------')
                    return 1.0 
                else:
                    return 0.0

            # elif q_type == "qa_seg":
            # elif gt_response == "qa_seg":
            else:
                # 使用 difflib 做单词级模糊匹配
                gt = gt_response.lower()
                output_words = output_text.split()
                # if len(output_words) > 3:
                #     return 0.0  todo 最近一次训练还未加入 
                for word in output_words:
                    similarity = difflib.SequenceMatcher(None, word, gt).ratio()
                    if similarity >= 0.8:
                        print('-------------------------------open正确-------------------------------')
                        return 1.0
                return 0.0
    except Exception:
        pass  # Continue to next verification method if this fails
    return 0.0
    
def seg_strict_compute_score8_region_qa(predict_str: str, ground_truth: str) -> float:  # todo 传个type进来
    # print('ground_truth=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-{}'.format(ground_truth))
    # print('predict_str=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-{}'.format(predict_str))
    # print('--------------------------------reward6_qa——box--------------------------------')
    thinking_format_reward = seg_thinking_format_reward(predict_str)
    segmentation_format_reward = seg_segmentation_format_reward(predict_str)  # 加入qa对话模板
    # print('--------------------------------segmentation_format_reward--------------------------------',segmentation_format_reward)
    box_iou_reward = seg_iou_reward(predict_str, ground_truth)
    # point_l1_reward = seg_point_l1_reward(predict_str, ground_truth)
    box_l1_reward = seg_box_l1_reward(predict_str, ground_truth)
    text_reward = compute_text_reward(predict_str, ground_truth)
    # logger.info('--------------------------------text_reward--------------------------------', text_reward)

    # box_in_box_and_size_reward = seg_box_in_box_and_size_reward(predict_str, ground_truth)
    box_in_box_only_reward = seg_box_in_box_only_reward(predict_str, ground_truth)
    box_size_only_reward = seg_box_size_only_reward(predict_str)
    # reward = iou_reward + thinking_format_reward + segmentation_format_reward + point_l1_reward + box_l1_reward # original 
    # reward = box_in_box_and_size_reward + thinking_format_reward + segmentation_format_reward + point_l1_reward 
    # reward = thinking_format_reward + text_reward + segmentation_format_reward + box_l1_reward + box_iou_reward + box_in_box_and_size_reward # 左 前
    reward = thinking_format_reward + text_reward + segmentation_format_reward + box_l1_reward + box_iou_reward + box_in_box_only_reward + box_size_only_reward # 右 后
    scores = {
        "thinking_format_reward": thinking_format_reward,
        "text_reward": text_reward,
        "segmentation_format_reward": segmentation_format_reward,
        "box_l1_reward": box_l1_reward,
        "box_iou_reward": box_iou_reward,
        "box_in_box_only_reward": box_in_box_only_reward,
        "box_size_only_reward": box_size_only_reward,
        # "point_l1_reward_offline": point_l1_reward_offline
    }
    # assert 1 == 2
    return reward, scores