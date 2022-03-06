<base target="_blank"/>


# Image Enhancement<br>

Here we provide PyTorch implementation and the pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Base Model with Guiding
- Download MIT-Adobe-5K-UPE dataset.

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base_gd --name base_gd_adobe5k_test --dataset_root <dataset_dir> --dataset_name Adobe5k --batch_size xx --init_port xxx
```
- Test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_gd --name base_gd_adobe5k_test --dataset_root <dataset_dir> --dataset_name Adobe5k --batch_size xx --init_port xxxx
```
- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1h9EG2kZnYi3GI4nAsqnJb1HHBv8GeNf7/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1mhAxHjetfIvZv-O-kqeHTA) (access code: 0r0k), and put `latest_net_G.pth` in the directory `checkpoints/base_gd_enhancement`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_gd --name base_gd_enhancement --dataset_root <dataset_dir> --dataset_name Adobe5k --batch_size xx --init_port xxxx
```
