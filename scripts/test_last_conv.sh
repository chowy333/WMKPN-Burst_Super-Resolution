#!/usr/bin/env bash
python -u test.py\
	--test_path './syn_burst_val'\
    	--model last_conv --bias --scale 4 --gpu_id 1\

# if run out of memory, lower batch_size down

