cd /home/tom/MOFA-Video/Training

EXP_NAME="train_stage1"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${EXP_NAME}_${timestamp}.log"
# set cuda visible device
export CUDA_VISIBLE_DEVICES=0
nohup accelerate launch train_stage1.py \
 --pretrained_model_name_or_path="./ckpts/stable-video-diffusion-img2vid-xt-1-1" \
 --output_dir="logs/${EXP_NAME}/" \
 --width=384 \
 --height=384 \
 --seed=42 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=5 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=2500 \
 --checkpoints_total_limit=100 \
 --validation_steps=2500 \
 --num_frames=25 \
 --gradient_checkpointing \
 --num_validation_images=4 \
 --sample_stride=4 > ${log_file} 2>&1 &

 cursor ${log_file}

 