# 🚀 Finers

Code for paper "FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning" Neurips2025.

------------------------------------------------------------------------

## 📦 Installation

``` bash
# Create environment
conda create -n finers python=3.10
conda activate finers

# Project requirements
pip install -r requirements.txt

# Install PyTorch (CUDA 11.8)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1     --index-url https://download.pytorch.org/whl/cu118

# xFormers
pip install -U xformers==0.0.29     --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install bitsandbytes accelerate loguru pycocotools matplotlib sam2
pip install flash-attn --no-build-isolation   # may take long, or download from GitHub releases

# Editable install
pip install -e .
```

------------------------------------------------------------------------

## 🤖 Download Model

``` bash
apt install git-lfs
mkdir ckpts && cd ckpts

git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

------------------------------------------------------------------------

## ▶️ Run

### 1️⃣ LR Data Processing

``` bash
python data_process/data_converter_fixed_1920_qa.py
```

------------------------------------------------------------------------

### 2️⃣ LR Training

``` bash
bash training_scripts/final_hr_training.sh
```

------------------------------------------------------------------------

### 3️⃣ HR Data Processing (Two Methods)

#### **3.1 Paper Method (Search-Based)**

``` bash
# Step 1: region search based on LR model
bash data_process/data_convert_1920_with_best_region_by_LR_model.sh

# Step 2: HR conversion
python data_process/data_converter_fixed_1920_qa_with_best_region.py
```

#### **3.2 Random Region Method**

``` bash
python data_process/data_converter_fixed_512_gt_crop_random_region.py
```

------------------------------------------------------------------------

### 4️⃣ HR Training

``` bash
bash training_scripts/final_lr_training.sh
```

------------------------------------------------------------------------

## 🔄 Model Conversion (HF Format)

``` bash
python3 training_scripts/model_merger.py     --local_dir workdir/xxx/global_step_xxx/actor
```

------------------------------------------------------------------------

## 🧪 Evaluation

``` bash
bash eval.sh
```

------------------------------------------------------------------------
## Acknowledgement
 - Our repo is built on [Seg-Zero](https://github.com/dvlab-research/Seg-Zero), [EasyR1](https://github.com/dvlab-research/Seg-Zero?tab=readme-ov-file) and [veRL](https://github.com/volcengine/verl). We thank the authors for sharing their codes.
 - This work utilizes models from  [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [SAM2](https://huggingface.co/facebook/sam2-hiera-large). 
