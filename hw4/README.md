# NYCU Visual Recognition using Deep Learning 2025 Spring HW4

StudentID: 111550066

Name: 王裕昕

## Introduction

The task is to restore images with rain or snow degradation. In the dataset, there are colored images with rain or snow degradation and their corresponding clean images. There are two folders under the dataset: train and test. The model is trained using the images under the ‘train’ folder. The goal is to maximize the PSNR when restoring the images under the ‘test’ folder. 

The model is based on the PromptIR architecture and trained from scratch, you can run the training code by:

```bash
python train.py --de_type=rain snow --data_file_dir path_to_your_dataset
```

or
```bash
CUDA_VISIBLE_DEVICES=1,2 python train.py --de_type=rain snow --data_file_dir path_to_your_dataset
```
if you want to specify certain GPU cards

to train the model with the data under the "train" folder. For more argument options, please refer to options.py. After training, there will be a directory "train_ckpt" where the checkpoints are saved. The checkpoints are saved every 10 epochs and when the validation PSNR reached a new best score.

The dataset is built with the class in utils/dataset_utils.py and preprocessed with utils/dataset_utils.py and utils/image_utils.py. The scheduler applied during training is imported from utils/schedulers.py and the model used for training is net/model.py. 

After training, you can run
```bash
python testing.py
```
to generate restored images with the saved model checkpoint and the data under the "test" folder, then output the "pred.npz" with the required format.

## How to Install
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install torch, torchvision, tqdm, matplotlib, pillow, scikit-image
```bash
pip install torch==2.6.0 torchvision==0.21.0 tqdm==4.64.0 pillow==9.0.1 einops==0.8.1 numpy==1.21.5 lightning==2.5.1.post0 pytorch_msssim==1.0.0
```
or by
```bash
pip install -r requirements.txt
```

## Performance Snapshot
![alt text](leaderboard_snapshot.png)
