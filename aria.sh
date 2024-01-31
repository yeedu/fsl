%%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.
model_root='./model/' #'/mnt/data1/llr_data/AMD_Classification/model/'
data_path='/data/yedu/FSL/AMD_Classification/fewshot'
output_dir='./vpt_prompt/aria_shallow/'
 


for i in "1"; do       
    for seed in "42" "44" "82" "100" "800"; do
        CUDA_VISIBLE_DEVICES=4 python train.py \
            --config-file configs/prompt/aria.yaml \
            MODEL.TYPE "vit" \
            DATA.BATCH_SIZE "2" \
            MODEL.PROMPT.NUM_TOKENS "100" \
            MODEL.PROMPT.DEEP "False" \
            MODEL.PROMPT.DROPOUT "0.1" \
            DATA.FEATURE "sup_vitb16_imagenet21k" \
            DATA.NUMBER_CLASSES "2" \
            SOLVER.BASE_LR "0.25" \
            SOLVER.WEIGHT_DECAY "0.001" \
            SEED ${seed} \
            MODEL.MODEL_ROOT "${model_root}" \
            DATA.DATAPATH "${data_path}" \
            OUTPUT_DIR "${output_dir}/test${i}/seed${seed}" \
            DATA.num "${i}"
    done
    
done


for i in "2" "4" "8" "16"; do       
    for seed in "42" "44" "82" "100" "800"; do
        CUDA_VISIBLE_DEVICES=4 python train.py \
            --config-file configs/prompt/aria.yaml \
            MODEL.TYPE "vit" \
            DATA.BATCH_SIZE "2" \
            MODEL.PROMPT.NUM_TOKENS "100" \
            MODEL.PROMPT.DEEP "False" \
            MODEL.PROMPT.DROPOUT "0.1" \
            DATA.FEATURE "sup_vitb16_imagenet21k" \
            DATA.NUMBER_CLASSES "2" \
            SOLVER.BASE_LR "0.25" \
            SOLVER.WEIGHT_DECAY "0.001" \
            SEED ${seed} \
            MODEL.MODEL_ROOT "${model_root}" \
            DATA.DATAPATH "${data_path}" \
            OUTPUT_DIR "${output_dir}/test${i}/seed${seed}" \
            DATA.num "${i}"
    done
done