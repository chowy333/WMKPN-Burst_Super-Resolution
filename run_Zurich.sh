#!/usr/bin/env bash
python -u main.py >> footprint.txt\
	--train_path '/home/wooyeong/Burst/burstsr_dataset/Zurich_Public/'\
    --model ours --bias --scale 4\
    --max_noise 0.078 --min_noise 0.00\

# if run out of memory, lower batch_size down

