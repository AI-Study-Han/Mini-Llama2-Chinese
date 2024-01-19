lr=2e-4


config_name=./config.json

chinese_tokenizer_name=THUDM/chatglm3-6b


per_device_train_batch_size=16
gradient_accumulation_steps=1
block_size=512
output_dir=output_dir/model9

deepspeed_config_file=ds_zero2_no_offload.json
random_seed=$RANDOM

torchrun --nnodes 1 --nproc_per_node 8 run_clm_pt_with_peft_modify.py \
    --deepspeed ${deepspeed_config_file} \
    --config_name ${config_name} \
    --tokenizer_name ${chinese_tokenizer_name} \
    --validation_split_percentage 0.0001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed 1234 \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
