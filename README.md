# FinersV2

## 安装

```bash
conda create -n finers python=3.10

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install bitsandbytes accelerate loguru
pip install flash-attn --no-build-isolation  # 会很慢 可以去这里下载https://github.com/Dao-AILab/flash-attention/releases

pip3 install -U xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu118

pip install loguru pycocotools matplotlib sam2

pip install -r requirements.txt

pip install -e .
```

## 模型

```bash
# Qwen2.5
apt install git-lfs
mkdir ckpts
cd ckpts
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct 
```

## 运行

### 1. LR数据处理

low resolution

```bash
python data_process/data_converter_fixed_1920_qa.py
```

### 2. LR 训练

```bash
bash training_scripts/final_hr_training.sh
```

## 3. HR数据处理 （两个方法）

### 3.1 paper method 

```bash
# 1. search
bash data_process/data_convert_1920_with_best_region_by_LR_model.sh

# 2. convert
python data_process/data_converter_fixed_1920_qa_with_best_region.py 
```

### 3.2 random method

```bash
python data_process/data_converter_fixed_512_gt_crop_random_region.py
```

### 4. HR 训练

```bash
bash training_scripts/final_lr_training.sh
```

## 5. 模型转换

训练好的LR和HR模型要转换为 HuggingFace 格式才可以做推理评估：

```bash
python3 training_scripts/model_merger.py --local_dir workdir/xxx/global_step_xxx/actor
```

## 6. 评估

Reward8 没有 cross stage 时与 random LR 联合都带有 QA 训练后测试：

```bash
bash eval.sh
```
