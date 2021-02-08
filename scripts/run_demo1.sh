#!/usr/bin/env bash
python -u main.py >> footprint_demo1-8-64-x4.txt\
	--train_path '/home/samsung/wycho/datasets/Zurich_Public/'\
    --model demo1 --bias --scale 4 --gpu_id 1\


# if run out of memory, lower batch_size down

