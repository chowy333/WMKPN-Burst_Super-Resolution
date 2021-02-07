#!/usr/bin/env bash
python -u main.py >> footprint_demo5-8-64-x4.txt\
	--train_path '/home/wooyeong/Burst/burstsr_dataset/Zurich_Public/'\
    --model demo5 --bias --scale 4 --gpu_id 3\
  

# if run out of memory, lower batch_size down

