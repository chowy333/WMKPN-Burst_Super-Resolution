# Ntire2021

## Updated

I downloaded every dataset to "/home/wooyeong/Burst/dataset/"

## Dates
* 2021.01.26 Release of train and validation data  
* 2021.02.01 Validation server online  
* 2021.03.01 Final test data release (inputs only)  
* 2021.03.08 Test output results submission deadline  
* 2021.03.09 Fact sheets and code/executable submission deadline  
* 2021.03.11 Preliminary test results released to the participants  
* 2021.03.28 Paper submission deadline for entries from the challenge  
* 2021.06.15 NTIRE workshop and challenges, results and award ceremony (CVPR 2021, Online)  


## Introduction

* The task of this challenge is to generate a denoised, demosaicked, higher-resolution image, given a RAW burst as input

* The challenge uses a new dataset and has 2 tracks, namely Track 1: Synthetic and Track 2: Real-world.

## Dataset

* 300장의 GT RGB, 각각 14개의 sequenced bayer Raw image(RGGB)로 구성되어있다.
* The images in the burst have unknown offsets with respect to each other, and are corrupted by noise.
* The goal is to exploit the information from the multiple input images to predict a denoised, demosaicked RGB image having a 4 times higher resolution, compared to the input. (scale 4배의 성능을 내는 것이 목표)
* test split of the Zurich RAW로 부터 

## Track 1 - Synthetic

* HR RGB 이미지 부터 LR input burst를 만들어내는 pipeline

**Data generation** : sRGB 이미지를 inverse ISP하고, Random tranlation과 rotation으로 LR burst를 만든 후 , Bilinear downsampling함. 그 이후, mosaic 해주고 noise 첨가

**Training set** : synthetic burst만드는 코드 제공, Zurich dataset빼고 다 적용가능

**Validation set** : The bursts in the validation set have been pre-generated with the data generation code, using the RGB images from the test split of the Zurich RAW to RGB dataset.

## Evaluation

*  the PSNR computation will be computed in the linear sensor space, before post-processing steps such as color correction, white-balancing, gamma correction etc.

## Submission

**Validation set** : Colab server, 2월 1일 open. save_results_synburst_val.py 확인해서 제출

---

## Track2 - Real-world

**BurstSR dataset** : 14 sequence의 handheld smartphone camera. crop도 맞춤. detailed specific 내용 곧 올라올 Deep Burst Super-Resolution 참조.

**Challenges** : Burst 이미지간에 mis-alignment를 맞추는 것이 keypoint.

**Evaluation** : AlignedPSNR => AlignedPSNR first spatially aligns the network prediction to the ground truth, using pixel-wise optical flow estimated using PWC-Net. A linear color mapping between the input burst and the ground truth, modeled as a 3x3 color correction matrix, is then estimated and used to transform the spatially aligned network prediction to the same color space as the ground truth. Finally, PSNR is computed between the spatially aligned and color corrected network prediction and the ground truth. More description of the AlignedPSNR metric is available in the paper "Deep Burst Super-Resolution" (link will be posted soon).

**User Study** : original을 최대한 잘 복원하는 것이 목표이다.

## Submission

* No evaluation server. validation set의 GT와 평가 지표 제공해줄꺼임



