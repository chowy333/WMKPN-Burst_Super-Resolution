#!/usr/bin/env bash
python -u main.py >> footprint_demo2-8-64-x4.txt\
	--train_path '/home/samsung/wycho/datasets/Zurich_Public/'\
    --model demo2 --bias --scale 4 --gpu_id 2\
    --sr_lambda 0 --ssim_lambda 1


# if run out of memory, lower batch_size down

