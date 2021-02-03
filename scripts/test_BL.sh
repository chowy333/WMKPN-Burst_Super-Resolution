#!/usr/bin/env bash
python -u test.py >> footprint_BL-x4.txt\
	--test_path '/home/wooyeong/Burst/burstsr_dataset/Zurich_Public/'\
    	--model last_conv --bias --scale 4 --gpu_id 1\

# if run out of memory, lower batch_size down

