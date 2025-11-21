# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import re
import json
import cv2
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl import DataProto

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index
from tensordict import TensorDict
from verl import DataProto


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        unified_qa_model: False,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
    ):
        self.unified_qa_model = unified_qa_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        self.dataset = load_dataset(data_path, cache_dir="data_cache")['train']
        # self.dataset = load_from_disk(data_path)['train'] # you can load from disk if you have already downloaded the dataset
                
        
        if unified_qa_model: 
            # # new prompt
            # self.user_prompt_referring = "<image>" \
            #                 "Please find '{Question}' with an approximate 256*256 bbox." \
            #                 "Compare the difference between objects and find the most closely matched one." \
            #                 "Output the thinking process in <think> </think>" \
            #                 "Output the 256*256 bbox and the final response in JSON format." \
            #                 "i.e., <think> thinking process here </think>" \
            #                 "<answer>{Answer}</answer>"
            # self.user_prompt_qa = "<image>" \
            #                 "Please find '{Question}' with an approximate 256*256 bbox." \
            #                 "Compare the difference between objects and find the most closely matched one." \
            #                 "Output the thinking process in <think> </think>" \
            #                 "Output the 256*256 bbox and the final response in JSON format." \
            #                 "i.e., <think> thinking process here </think>" \
            #                 "<answer>{Answer}</answer>"
            # self.user_prompt_qa_option = "<image>" \
            #                 "Please find '{Question} {Options}' with an approximate 256*256 bbox." \
            #                 "Compare the difference between objects and find the most closely matched one." \
            #                 "Output the thinking process in <think> </think>" \
            #                 "Output the 256*256 bbox and the final response in JSON format." \
            #                 "i.e., <think> thinking process here </think>" \
            #                 "<answer>{Answer}</answer>"
            # old prompt
            self.user_prompt_referring = "<image>" \
                "Based on the '{Question}', identify a 256*256 bounding box that best localizes the region most relevant to the query. And respond with whether the object is found." \
                "Compare the difference between regions and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the 256*256 region bbox and the final response inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
            self.user_prompt_qa = "<image>" \
                "Based on the '{Question}', identify a 256*256 bounding box that best localizes the region most relevant to the query. And give me a final response with a word or phrase." \
                "Compare the difference between regions and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the 256*256 region bbox and the final response inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
            self.user_prompt_qa_option = "<image>" \
                "Based on the '{Question}', identify a 256*256 bounding box that best localizes the region most relevant to the query. And give me a correct option from {Options}." \
                "Compare the difference between regions and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the 256*256 region bbox and the final response inside the interested object in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
            
            
            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
            # For reward 6 7 yjz
            # self.user_prompt_referring = "<image>" \
            #     "Please find '{Question}' with bbox and points. And respond with whether the object is found." \
            #     "Compare the difference between objects and find the most closely matched one." \
            #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            #     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
            #     "i.e., <think> thinking process here </think>" \
            #     "<answer>{Answer}</answer>"
            # self.user_prompt_qa = "<image>" \
            #     "Please find '{Question}' with bbox and points. And give me a final response with a word or phrase." \
            #     "Compare the difference between objects and find the most closely matched one." \
            #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            #     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
            #     "i.e., <think> thinking process here </think>" \
            #     "<answer>{Answer}</answer>"
            # self.user_prompt_qa_option = "<image>" \
            #     "Please find '{Question}' with bbox and points. And give me a correct option from {Options}." \
            #     "Compare the difference between objects and find the most closely matched one." \
            #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            #     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
            #     "i.e., <think> thinking process here </think>" \
            #     "<answer>{Answer}</answer>"
        else:
            self.user_prompt = "<image>" \
                "Based on the '{Question}', identify a 256*256 bounding box that best localizes the region most relevant to the query." \
                "Compare the difference between regions and find the most closely matched one." \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                "Output the 256*256 region bbox in JSON format." \
                "i.e., <think> thinking process here </think>" \
                "<answer>{Answer}</answer>"
            


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]
        # print('row_dict=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-', row_dict.keys())
        # dict_keys(['id', 'problem', 'solution', 'answer', 'type', 'image', 'options'])
        if self.unified_qa_model:
            # For reward 6 7 yjz 上面同理
            if row_dict["type"]== "referring":
                messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_referring.format(Question=row_dict["problem"].lower().strip("."),
                                                                    Answer="{'bbox': [x_min,y_min,x_min+256,y_min+256], 'response': 'The object is found.'}")},
            ]
            elif row_dict["type"]== "option":
                assert row_dict["options"], "Options should not be empty!"
                messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_qa_option.format(Question=row_dict["problem"].lower().strip("."),
                                                                    Answer="{'bbox': [x_min,y_min,x_min+256,y_min+256], 'response': 'A'}",
                                                                    Options=row_dict["options"])},
            ]
            else:
                messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_qa.format(Question=row_dict["problem"].lower().strip("."),
                                                                    Answer="{'bbox': [x_min,y_min,x_min+256,y_min+256], 'response': 'The object is white.' }")},
            ]
             # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
            # For reward 6 7 yjz

            # if row_dict["type"]== "referring":
            #     messages = [
            #     {"role": "system", "content": self.system_prompt},
            #     {"role": "user", "content": self.user_prompt_referring.format(Question=row_dict["problem"].lower().strip("."),
            #                                                         Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180], response: 'The object is found.'}")},
            # ]
            # elif row_dict["type"]== "option":
            #     assert row_dict["options"], "Options should not be empty!"
            #     messages = [
            #     {"role": "system", "content": self.system_prompt},
            #     {"role": "user", "content": self.user_prompt_qa_option.format(Question=row_dict["problem"].lower().strip("."),
            #                                                         Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180], response: 'A'}",
            #                                                         Options=row_dict["options"])},
            # ]
            # else:
            #     messages = [
            #     {"role": "system", "content": self.system_prompt},
            #     {"role": "user", "content": self.user_prompt_qa.format(Question=row_dict["problem"].lower().strip("."),
            #                                                         Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180], response: 'The object is white.' }")},
            # ]
            # # print('messages=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-',messages)
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(Question=row_dict["problem"].lower().strip("."),
                                                                    Answer="{'bbox': [x_min,y_min,x_min+256,y_min+256]")},
            ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["images"] = [
                process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict["id"] = [row_dict["id"]]
        row_dict["problem"] = [row_dict["problem"]]
        # print(row_dict["id"])
        # print(row_dict["problem"])
        return row_dict

    def extract_predicted_box(self, predict_str:str, x_factor, y_factor):
        try:
            json_pattern = r'{[^}]+}'  
            json_match = re.search(json_pattern, predict_str)
            # pdb.set_trace()
            if json_match:
                data = json.loads(json_match.group(0))
                bbox_key = 'bbox'
                if bbox_key and len(data[bbox_key]) == 4:
                    content_bbox = data[bbox_key]
                    content_bbox = [round(int(content_bbox[0])*x_factor), round(int(content_bbox[1])*y_factor), round(int(content_bbox[2])*x_factor), round(int(content_bbox[3])*y_factor)]
                else:
                    content_bbox = None
            else:
                content_bbox = None
        except Exception:
            content_bbox = None
            
        return content_bbox
    
    def crop_image_around_gtbox(self, image_ori, gt_box, crop_size=512):
        """
        Args:
            image_ori: 原图
            gt_box: 原图上的GT框 [x_min, y_min, x_max, y_max]
            crop_size: 裁剪图像大小（默认512）
        
        Returns:
            cropped_img: 裁剪后的图像 (512x512 RGB)
            new_gt_box: 映射到裁剪图中的新GT框坐标 [x_min, y_min, x_max, y_max]
        """
        h, w, _ = image_ori.shape

        # 获取GT box中心点
        x_min, y_min, x_max, y_max = gt_box
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # 计算裁剪区域的左上角坐标
        half_crop = crop_size // 2
        x1 = max(cx - half_crop, 0)
        y1 = max(cy - half_crop, 0)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # 如果裁剪区域超出边界，则回退坐标
        if x2 > w:
            x2 = w
            x1 = max(w - crop_size, 0)
        if y2 > h:
            y2 = h
            y1 = max(h - crop_size, 0)

        # 裁剪图像
        cropped_img = image_ori[y1:y2, x1:x2]

        # 映射GT box坐标到裁剪图像坐标
        new_x_min = x_min - x1
        new_y_min = y_min - y1
        new_x_max = x_max - x1
        new_y_max = y_max - y1
        new_gt_box = [new_x_min, new_y_min, new_x_max, new_y_max]

        return cropped_img, new_gt_box

    def update_batch_for_lr(self, batch_data: DataProto) -> DataProto:
        system_prompt = r"You are a helpful assistant."
        user_prompt = "<image>" \
            "Please find '{Question}' with bbox and points." \
            "Compare the difference between objects and find the most closely matched one." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"
        
        for i in range(len(batch_data)):
            data_item = batch_data[i]  # DataProtoItem
            
            # data_type = data_item.batch["type"]
            # non_tensor_batch = data_item.non_tensor_batch
            image_name = data_item.non_tensor_batch["id"][0]
            problem = data_item.non_tensor_batch["problem"][0]
            
            print("len problem:{}".format(len(data_item.non_tensor_batch["problem"])))
            print("len image_name:{}".format(len(data_item.non_tensor_batch["id"])))
            print("problem:{}".format(problem))
            print("image_name:{}".format(image_name))
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            print("response_str:{}".format(response_str))
                    
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(Question=problem.lower().strip("."),
                                                                Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}")},
            ]
                
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            # image_for_hr = data_item.batch["images"]
            image_path = image_name + ".jpg"
            image_path = os.path.join("all_data/all_images", image_path)
            assert os.path.exists(image_path)
            print(image_path)
            image_np = cv2.imread(image_path)
            image_ori = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
            original_width, original_height = image.size
            x_factor, y_factor = original_width/512, original_height/512
            
            predicted_bbox = self.extract_predicted_box(response_str, x_factor, y_factor) # size不对
            print("predicted_bbox:{}".format(predicted_bbox))
            
            if predicted_bbox != None:
                cropped_img, _ = self.crop_image_around_gtbox(image_ori, predicted_bbox, crop_size=512)
                print("cropped_img:{}".format(cropped_img.shape))
                cropped_img = Image.fromarray(cropped_img)
                
                raw_prompt = prompt
                
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=prompt,
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
                
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)
                
                data_item.non_tensor_batch["cropped_img"] = [cropped_img]
                data_item.batch["input_ids_lr"] = input_ids
                data_item.batch["attention_mask_lr"] = attention_mask
                data_item.batch["position_ids_lr"] = position_ids
                data_item.batch["raw_prompt_ids_lr"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            else:
                data_item.non_tensor_batch["cropped_img"] = None
                
        batch_size = batch_data.batch.batch_size[0]
        print("batch size:{}".format(batch_size))
        batch_data.__post_init__()
        # return DataProto(batch=batch_data.batch, non_tensor_batch=batch_data.non_tensor_batch)
        # batch = TensorDict(
        #     {
        #         "prompts": input_ids,
        #         "responses": response_ids,
        #         "input_ids": sequence_ids,  # here input_ids become the whole sentences
        #         "attention_mask": attention_mask,
        #         "position_ids": position_ids,
        #     },
        #     batch_size=batch_size,
        # )
        # return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
        # batch = TensorDict(
        #     {
        #         "prompts": input_ids,
        #         "responses": response_ids,
        #         "input_ids": sequence_ids,  # here input_ids become the whole sentences
        #         "attention_mask": attention_mask,
        #         "position_ids": position_ids,
        #     },
        #     batch_size=batch_size,
        # )
        # return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            # continue

            # row_dict["input_ids"] = input_ids
            # row_dict["attention_mask"] = attention_mask
            # row_dict["position_ids"] = position_ids
            # row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)