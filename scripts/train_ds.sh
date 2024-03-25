torchrun --nproc_per_node=4 --master_port=20001 train.py \
    --model_name_or_path path/to/your/model  \
    --data_path path/to/your/data \
    --bf16 True \
    --output_dir path/to/output/model \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --deepspeed 'stanford_alpaca/configs/default_offload_opt_param.json' \
