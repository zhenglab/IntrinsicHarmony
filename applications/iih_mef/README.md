<base target="_blank"/>


# Multi-Exposure Image Fusion<br>

Here we provide PyTorch implementation and the pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Base Model with Guiding
- Download SICE dataset.

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base_gd  --name base_gd_sice_test --dataset_root <dataset_dir> --dataset_name mef  --batch_size xx --init_port xxxx
```
- Test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_gd  --name base_gd_sice_test --dataset_root <dataset_dir> --dataset_name mef  --batch_size xx --init_port xxxx
```
- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/17SIkVhRFW5LTuX2PXDPkVw2IwKWDpO-B/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1V4ulhcC1eqM6EfVbxRIz1g) (access code: 15vn), and put `latest_net_G.pth` in the directory `checkpoints/base_gd_mef`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_gd  --name base_gd_mef --dataset_root <dataset_dir> --dataset_name mef  --batch_size xx --init_port xxxx
```