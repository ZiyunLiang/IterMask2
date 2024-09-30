# $IterMask^2$: Iterative Unsupervised Anomaly Segmentation via Spatial and Frequency Masking for Brain Lesions in MRI"</a>, Multiscale Multimodal Medical Imaging - MICCAI, 2024

This repository is the official pytorch implementation for paper: Liang et al., <a href="https://arxiv.org/abs/2308.16150"> IterMask2: Iterative Unsupervised Anomaly Segmentation via Spatial and Frequency Masking for Brain Lesions in MRI"</a>, - MICCAI, 2024 (Oral).


## Introduction:
Unsupervised anomaly segmentation approaches to pathology segmentation train a model on images of healthy subjects, that they define as the 'normal' data distribution. At inference, they aim to segment any pathologies in new images as 'anomalies', as they exhibit patterns that deviate from those in 'normal' training data.
Prevailing methods follow the 'corrupt-and-reconstruct' paradigm. They intentionally corrupt an input image, reconstruct it to follow the learned 'normal' distribution, and subsequently segment anomalies based on reconstruction error. Corrupting an input image, however, inevitably leads to suboptimal reconstruction even of normal regions, causing false positives. 
To alleviate this, we propose a novel iterative spatial mask-refining strategy $\rm{IterMask^2}$. 
We iteratively mask areas of the image, reconstruct them, and update the mask based on reconstruction error. This iterative process progressively adds information about areas that are confidently normal as per the model. The increasing content guides reconstruction of nearby masked areas, improving reconstruction of normal tissue under these areas, reducing false positives. 
We also use high-frequency image content as an auxiliary input to provide additional structural information for masked areas. This further improves reconstruction error of normal in comparison to anomalous areas, facilitating segmentation of the latter. 


[//]: # 
![Image text](https://github.com/ZiyunLiang/IterMask2/blob/main/img/img1.png)

[//]: # (![Image text]&#40;https://github.com/ZiyunLiang/Itermask2/img/img2.png&#41;)

## Usage:

### 1. preparation
**1.1 Environment**
We recommand you using conda for installing the depandencies.
The following command will help you create a new conda environment will all the required libraries installed: 
```
conda env create -f environment.yml
conda activate IterMask2
```
For manualy installing packages:
- `Python`                 3.9
- `torch`                   2.3.1
- `blobfile`                2.1.1
- `numpy`                   1.26.3
- `scikit-image`            0.24.0
- `scipy`                   1.13.1
- `monai`                   1.3.1
- `tensorboard`            2.17.0
- `ml_collections`         0.1.1
- `torchmetrics`             1.4.0

The project can be cloned using:
```
https://github.com/ZiyunLiang/IterMask2.git
```
**1.2 Dataset Download**\
Download the BraTS2021 training data from <a href="http://www.braintumorsegmentation.org/">BraTS2021 official website</a> and unzip it to the folder `./datasets/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021`.
The `brats_split_training.txt`, `brats_split_validation.txt`, and `brats_split_testing.txt` in `./dataset` are the split of the dataset used in the paper. 

**1.3 Data Preprocessing**\
We preprocess the 3D data and save the normalized 2D slices for training to save the data loading time.
We have the data preprocessing script in `./datasets/brats_preprocess.py`. And we present an example of splitting the dataset into brats_split_training.txt, brats_split_validation.txt, and brats_split_testing.txt
From every 3D image, we extract slices 70 to 90, that mostly capture the central part of the brain. For training, we only use slices that
do not contain any tumors. And for testing subjects, the slide with the biggest tumor is selected. Data is normalized during the preprocessing process. The (percentage_upper, percentage_lower) intensity of the image is normalized to [-3,3], and you can change which percentage to normalize in the command line argument.
The command line arguments for this script are:
  - `--modality` Allows you to choose which modality you want to preprocess and save. Multiple modalities can be added, but should be separated with ',' without space. Default: 't2,flair,t1'
  - `--data_dir` The directory of the already downloaded brats data. Default: './datasets/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
  - `--output_dir` The directory to save the preprocessed data. Default: './data/BRATS'
  - `--percentage_upper` When normalizing the data, the upper percentage of data intensity for normalization, the value should be between[0-1]. Default: 0.9
  - `--percentage_lower` When normalizing the data, the lower percentage of data intensity for normalization, the value should be between[0-1]. Default: 0.02

Below is an example script for preprocessing data (all arguments are set to default).
```
python ./datasets/brats_preprocess.py 
```

### 2. Training

The training script is in `train.py`. The arguments for this script are:
  - `--gpu_id` Allows you to choose the GPU that you want to use for this experiment. Default: '0'
  - `--dataset` Allows you to choose the dataset that you want to use for this experiment. Feel free to test it on your own dataset. Default: 'brats'
  - `--data_dir` The directory of the already preprocessed brats data. Default: './datasets/data'
  - `--modality` The input modality to the model. Default: 'flair'
  - `--experiment_name` The file name for saving the model. Default: 'None'
  - `--model_name` The model to train: first_iter or masked_autoencoder. (First iteration is trained separately.) Default: 'masked_autoencoder'

The other hyperparameters used for training are in `./config/brats_config.py`. 
Below is an example script for training the model. Note that the model need to be trained twice 
because the first iteration need to be trained separately. (all other arguments are set to default).
```
python train.py --model_name first_iter
python train.py --model_name masked_autoencoder
```

### 3. Testing 
The testing script is in `test.py`.
The arguments for this script are:
  - `--gpu_id` Allows you to choose the GPU that you want to use for this experiment. Default: '0'
  - `--dataset` Allows you to choose the dataset that you want to use for this experiment. Feel free to test it on your own dataset. Default: 'brats'
  - `--modality` Input modality, choose from flair, t2, t1, t1ce. Default: 'flair'
  - `--data_dir` The directory of the already preprocessed brats data. Default: './datasets/data'
  - `--experiment_name_first_iter` The file name for trained first iteration model so that we can load the saved model. Default: 'None'
  - `--experiment_name_masked_autoencoder` The file name for trained masked autoencoder model so that we can load the saved model. Default: 'None'
  - `--best_threshold` whether to compute the result using the best threshold or not. Default: 'False'
  
The other hyperparameters used for testing are in `./config/brats_config.py`.

Below is an example script for training the model with our default settings:
```
python test.py
```

## Citation
If you have any questions, please contact Ziyun Liang (ziyun.liang@eng.ox.ac.uk) and I am happy to discuss more with you. 
If you find this work helpful for your project, please give it a star and a citation. 
We greatly appreciate your acknowledgment.
```
@article{liang2024itermask2,
  title={IterMask2: Iterative Unsupervised Anomaly Segmentation via Spatial and Frequency Masking for Brain Lesions in MRI},
  author={Liang, Ziyun and Guo, Xiaoqing and Noble, J Alison and Kamnitsas, Konstantinos},
  journal={arXiv preprint arXiv:2406.02422},
  year={2024}
}
```

## License
This project is licensed under the terms of the MIT license.
MIT License

Copyright (c) 2025 Ziyun Liang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
