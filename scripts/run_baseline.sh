#!/usr/bin/env bash
python -u main.py >> footprint_baseline-13-64-x4.txt\
	--train_path '/home/samsung/wycho/datasets/Zurich_Public/'\
    --model baseline --bias --scale 4 --gpu_id 2\
    --num_rfab 13 --sr_lambda 0.5 --ssim_lambda 0.5\


# if run out of memory, lower batch_size down

