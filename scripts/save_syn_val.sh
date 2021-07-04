#!/usr/bin/env bash
python -u save_results_synburst_val.py\
	--test_path './syn_burst_val' --save_path "val_out_last_conv"\
    	--model last_conv --pretrained_model "./ckpts/last_conv-8-64-x4_model_best.path" --bias --scale 4 --gpu_id 0\

# if run out of memory, lower batch_size down

