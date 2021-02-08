#!/usr/bin/env bash
python -u main.py >> footprint_demo3-8-64-x4.txt\
	--train_path '/home/samsung/wycho/datasets/Zurich_Public/'\
    --model demo3 --bias --scale 4 --gpu_id 3\
    --sr_lambda 0.5 --ssim_lambda 0.5\


# if run out of memory, lower batch_size down

