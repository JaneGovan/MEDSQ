export CUDA_LAUNCH_BLOCKING=1

MODE=MiniCPM-2B-sft-bf16
OUTPUT_PATH=./adapter_model
TRAIN_DATA=data/train_data.json
EVAL_DATA=data/eval_data.json

deepspeed --include localhost:0 models/scripts/finetune.py \
    --model_name_or_path $MODE \
    --output_dir $OUTPUT_PATH \
    --train_data_path $TRAIN_DATA \
    --eval_data_path $EVAL_DATA \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1  \
    --model_max_length 512 \
    --bf16 --use_lora \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --num_train_epochs 2 \
    --weight_decay 0.01 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --seed 42 \
    --log_level error \
    --logging_strategy steps \
    --logging_steps 100 \
    --deepspeed models/scripts/config_deepspeed/zero3.json
