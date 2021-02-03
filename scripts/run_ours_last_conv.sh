#!/usr/bin/env bash
python -u main.py >> footprint_last_conv-8-64-x4.txt\
	--train_path '/home/wooyeong/Burst/burstsr_dataset/Zurich_Public/'\
    --model last_conv --bias --scale 4 --gpu_id 3\
    --max_noise 0.078 --min_noise 0.00\

# if run out of memory, lower batch_size down

