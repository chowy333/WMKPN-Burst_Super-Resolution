# WMKPN

### Weighted Multi Kernel Prediction Network for Burst Image Super-Resolution

## Quick Test
### Dependencies
- Python >= 3
- [PyTorch 1.7.0](https://pytorch.org/) (CUDA version 10.1) 
- Python packages:  `pip install opencv-python Pillow scikit-image tensorboard`

### Test Models

- test with pretrained model by synthtic datasets
   1. check the test image path and output image path in the bash file
   2. set the path of pretrained ckpt 
   3. run the code with bash
        ```
        sh test_WMKPN.sh 
        ```  

## How to Train
We trained our model on synthesis datasets.

 1. check the train image path
 2. run the code with bash
       ```
        sh run_WMKPN.sh 
       ```
 3. Training procedure will be printed in footprint_{model_name}-bs_{batch_size}-{filters}-x{scale}.txt
 4. logs and checkpoint file will be saved on "logs/{args.post}/", "ckpts/{args.post}/" respectively.       

