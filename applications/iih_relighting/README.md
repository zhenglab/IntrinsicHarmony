<base target="_blank"/>


# Portrait Relighting<br>

Here we provide PyTorch implementation and the pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Base Model with Lighting
- Download DPR dataset.

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base_lt  --name base_lt_relighting_test --dataset_root <dataset_dir> --dataset_name DPR  --batch_size xx --init_port xxxx
```
- Test
```bash
# SH-based relighting
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt  --name base_lt_relighting_test --dataset_root <dataset_dir> --dataset_name DPR  --batch_size xx --init_port xxxx
#Image-based relighting
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt  --name base_lt_relighting_test --relighting_action transfer --dataset_root <dataset_dir> --dataset_name DPR --dataset_mode dprtransfer --batch_size xx --init_port xxxx
```

- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/11yGZvo-gLDRyfnO0A6xuqPmDaPcMB1en/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1yrUZ2YkT2bY9ThfYn_gJAg) (access code: bjqb), and put `latest_net_G.pth` in the directory `checkpoints/base_lt_relighting`. Run:

```bash
# SH-based relighting
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt  --name base_lt_relighting --dataset_root <dataset_dir> --dataset_name DPR --batch_size xx --init_port xxxx
#Image-based relighting
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt  --name base_lt_relighting --relighting_action transfer --dataset_root <dataset_dir> --dataset_name DPR --dataset_mode dprtransfer --batch_size xx --init_port xxxx
```
