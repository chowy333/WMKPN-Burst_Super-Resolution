#!/usr/bin/env bash
python -u test.py\
	--test_path '/SSD/samsung/wycho/datasets/Zurich_Public/' --save_output_path "./output/WMKPN_attn"\
    	--model WMKPN --pretrained_model "./ckpts/first/WMKPN-bs_8-64-x4_model_best.path" --bias --scale 4 --gpu_id 0\
            --color --blind_est --sep_conv --channel_att --spatial_att

# if run out of memory, lower batch_size down
