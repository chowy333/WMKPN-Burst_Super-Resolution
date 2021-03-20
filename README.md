# WMKPN

### Weighted Multi Kernel Prediction Network for Burst Image Super-Resolution

## Quick Test
### Dependencies
- Python >= 3
- [PyTorch 1.7.0](https://pytorch.org/) (CUDA version 10.1) 
- Python packages:  `pip install opencv-python Pillow scikit-image tensorboard`

### Test Models

- evalutate on test dataset with pretrained model by synthtic datasets (+ gamma correctioned images)
   1. check the test image path and output image path
   2. set the path of pretrained ckpt 
   3. run the code with bash
        ```
        sh ./scripts/test_WMKPN.sh 
        ```  

- save the output image with pretrained model by synthtic datasets (for submission)
   1. check the test image path and output image path
   2. set the path of pretrained ckpt 
   3. run the code with bash
        ```
        sh ./scripts/save_syn_WMKPN.sh 
        ```  
        
## How to Train
We trained our model on synthesis datasets.

 1. check the train image path
 2. run the code with bash
       ```
        sh ./scripts/run_WMKPN.sh
        ```
 3. Training procedure will be printed in footprint_{model_name}-bs_{batch_size}-{filters}-x{scale}.txt
 4. logs and checkpoint file will be saved on "logs/{args.post}/", "ckpts/{args.post}/" respectively.   
