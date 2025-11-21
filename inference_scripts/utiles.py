import json
import os
from loguru import logger
from PIL import Image as PILImage


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

def reform_messgae(is_final_answer, input_question, question_type, options, image, resize):
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