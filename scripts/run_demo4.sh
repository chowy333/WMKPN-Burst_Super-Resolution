#!/usr/bin/env bash
python -u main.py >> footprint_demo4-8-64-x4.txt\
	--train_path '/home/wooyeong/Burst/burstsr_dataset/Zurich_Public/'\
    --model demo4 --bias --scale 4 --gpu_id 2\
  

# if run out of memory, lower batch_size down

