#!/usr/bin/env bash
source /opt/conda/etc/profile.d/conda.sh
conda activate seg_zero


# pip install loguru
json_save_path=/15324359926/seg/HD_Reasoning/All_Middle_Process/predict_box_for_training_set_record_best_region
parquet_save_path=/15324359926/seg/HD_Reasoning-main-yjz/All_final_data/all_parquet_with_predicted_box_by_LR_model/train_v4_defalut_scale512region_prediected_bbox

cd /15324359926/seg/HD_Reasoning/Seg-Zero
CUDA_VISIBLE_DEVICES=0 python inference_scripts/search_best_region_by_trained_lr_model_for_HR_training.py  \
        --cascade_reasoning_model_path /15324359926/seg/HD_Reasoning/Seg-Zero/workdir/1_final_data_run_qwen2_5_7b_4gpu_512size_LR_model_baseline_qa_reward_revised0508_random_region/global_step_279/actor/huggingface \
        --resize_size "(1920, 1080)" \
        --cascade_resize_size "(512, 512)" \
        --segmentation_model_path //15324359926/seg/ckpts/sam2-hiera-large/sam2_hiera_large.pt \
        --segmentation_config_path //15324359926/seg/ckpts/sam2-hiera-large/sam2_hiera_l.yaml \
        --test_json_path /15324359926/seg/HD_Reasoning-main-yjz/All_final_data/all_json/middle_process/all_annotations_final_train.json \
        --image_path /15324359926/seg/HD_Reasoning-main-yjz/All_final_data/all_images  \
        --save_results \
        --save_path ${json_save_path} \
        --dynamic_box \

