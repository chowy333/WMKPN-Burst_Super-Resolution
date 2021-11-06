# Weighted Multi Kernel Prediction Network for Burst Image Super-Resolution (CVPRW 2021) <a href="https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Cho_Weighted_Multi-Kernel_Prediction_Network_for_Burst_Image_Super-Resolution_CVPRW_2021_paper.html" target="_blank">[PDF]</a>
By [Wooyeong Cho](https://sites.google.com/view/wooyeongcho), [Sanghyeok Son](https://sites.google.com/view/sanghyeokson/%ED%99%88), [Dae-Shik Kim](https://scholar.google.com/citations?user=nd-UgBYAAAAJ&hl=en&oi=ao) 

This projects took participate in New Trends in Image Restoration and Enhancement workshop and challenges on image and video processing (Ntire 2021 challenge) in conjunction with CVPR 2021 

## Quick Test
### Dependencies
- Python >= 3
- [PyTorch 1.7.0](https://pytorch.org/) (CUDA version 10.1) 
- Python packages:  `pip install opencv-python Pillow scikit-image tensorboard`

### Datasets downloads
- Synthetic data: Note that any image dataset except the 
test split of the [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset) 
can be used to generate synthetic bursts for training. 

- Real-World data: containing 639 real-world bursts from [BurstSR](https://data.vision.ee.ethz.ch/bhatg/track2_test_set.zip)


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

### WMKPN
Our model mainly consists of two part, WMKPN and SRNet. WMKPN is a one of the key contribution of this paper, utilizing the features from kernel branch and weight branch in this model. Weighted Multiple Kernels are usd to predict an accmulated kernel which can lead to better alignment module for burst images.

![total_net1](https://user-images.githubusercontent.com/46465539/126039353-5cf58307-8e40-4cc3-8ef8-c8b62e512eb2.PNG)

## Results

### Results on Real Images
![qual_syn_total](https://user-images.githubusercontent.com/46465539/126039403-2f66d5e4-8696-48dc-98c8-52d1c6dbb68f.PNG)

### Citation 
Please cite the following paper if you feel WMKPN is useful to your research
```
@inproceedings{cho2021weighted,
  title={Weighted Multi-Kernel Prediction Network for Burst Image Super-Resolution},
  author={Cho, Wooyeong and Son, Sanghyeok and Kim, Dae-Shik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={404--413},
  year={2021}
}
```
