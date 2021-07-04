#!/usr/bin/env bash
python -u main.py >> footprint_WMKPN-bs_8-64-x4.txt\
	--train_path '../burstsr_dataset/Zurich_Public/'\
    --model WMKPN --multi_kernel_size "1,3,5,7" --bias --scale 4 --gpu_id "0,1" --batch_size 8 --filters 64\
    --split_ratio 0.97 --num_workers 16 --sep_conv --sr_lambda 0.8 --ssim_lambda 0.2

# if run out of memory, lower batch_size down

