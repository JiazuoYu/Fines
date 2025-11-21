


CUDA_VISIBLE_DEVICES=0,1 python inference_scripts/eval_hd_reasoning_seg_zero_cascade_qa_add_dist_for_region_prompt_cross_stage.py \
        --reasoning_model_path workdir/final_hr_training/global_step_2/actor/huggingface \
        --cascade_reasoning_model_path workdir/final_lr_training/global_step_2/actor/huggingface \
        --resize_size "(1920, 1080)" \
        --cascade_resize_size "(512, 512)" \
        --segmentation_model_path /ckpts/sam2-hiera-large/sam2_hiera_large.pt \
        --segmentation_config_path /ckpts/sam2-hiera-large/sam2_hiera_l.yaml \
        --test_json_path all_data/all_json/all_annotations_final_test_v5.json \
        --image_path all_data/all_images \
        --save_path all_middle_process/test_cascade_HR_reward_qa_with_cross_stage_lr_reward_qa_random_region_visualization_final_version \
        --dynamic_box \
        --qa_stage "stage1" \
        --save_results