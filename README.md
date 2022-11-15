<base target="_blank"/>


# Delving Deep into Intrinsic Image Harmonization

Here we provide the PyTorch implementation and pre-trained model of our latest version, if you require the code of our previous CVPR version (**["Intrinsic Image Harmonization"](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Intrinsic_Image_Harmonization_CVPR_2021_paper.pdf)**), please click the **[release version](https://github.com/zhenglab/IntrinsicHarmony/releases/tag/v1.0)**.
# Colab Notebook to test on HVIDIT dataset : https://colab.research.google.com/drive/1XPketnGO9SwI-2R_cLN2cqX9calSZrn7#scrollTo=HUkx4oQn8-GB
## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Datasets
- Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset, and our **HVIDIT** dataset [Google Drive](https://drive.google.com/file/d/1-pa_9BNgIkuR0j1gcCxh8GI3XSWZN0e7/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1DR600XJFhm8lqfHZ6mOU_A) (access code: akbi).

## **Base Model**

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base  --name iih_base_allihd_test  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base  --name iih_base_allihd_test  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/13KJfvTJVz1F_OpLGX-Q2V-gSBIDCXtij/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1IZjyNlHOhYl-0ew044sY1g) (access code: n4js), and put `latest_net_G.pth` in the directory `checkpoints/iih_base_allihd`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base  --name iih_base_allihd  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## **Base Model with Lighting**

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base_lt  --name iih_base_lt_allihd_test  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt  --name iih_base_lt_allihd_test  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1BtQ7mFY4IWdILLKC6vlO3w-58B6XQ7AY/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1FN5aEBaM7kVzQ0lOPgLzwQ) (access code: hqhw), and put `latest_net_G.pth` in the directory `checkpoints/iih_base_lt_allihd`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt  --name iih_base_lt_allihd  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## **Base Model with Guiding**

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base_gd --name iih_base_gd_allihd_test --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_gd --name iih_base_gd_allihd_test --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1L4SgUBLi5wCfDb0bNmB_qac7WSk6WUfM/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1gDNyGbgsTEyDaPanknmjjQ) (access code: nqrc), and put `latest_net_G.pth` in the directory `checkpoints/iih_base_gd_allihd`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_gd --name iih_base_gd_allihd --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## **Base Model with Lighting and Guiding**

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base_lt_gd  --name iih_base_lt_gd_allihd_test  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt_gd  --name iih_base_lt_gd_allihd_test  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1ZyQGndcR2lC29CR2zQvrC-0KxjffYpF-/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1PM6vSgPyTgHE1eoKRPUdBQ) (access code: kmgp), and put `latest_net_G.pth` in the directory `checkpoints/iih_base_lt_gd_allihd`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt_gd  --name iih_base_lt_gd_allihd  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## **Base Model with Lighting and Guiding on iHarmony4 and HVIDIT Datasets**

- Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model iih_base_lt_gd  --name iih_base_lt_gd_newihd_test  --dataset_root <dataset_dir> --dataset_name newIHD --batch_size xx --init_port xxxx
```
- Test
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt_gd  --name iih_base_lt_gd_newihd_test  --dataset_root <dataset_dir> --dataset_name newIHD --batch_size xx --init_port xxxx
```
- Apply pre-trained model

Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1kG2LOKDlK_4FFLFH_q2_8BRHutUqkPfs/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1l9vS67O9CiWLiHwNlF_kfg) (access code: jnhg), and put `latest_net_G.pth` in the directory `checkpoints/iih_base_lt_gd_allihd`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model iih_base_lt_gd  --name iih_base_lt_gd_newihd  --dataset_root <dataset_dir> --dataset_name newIHD --batch_size xx --init_port xxxx
```

## Evaluation
We provide the code in `ih_evaluation.py`. Run:
```bash
# iHarmony4 dataset
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root  results/experiment/test_latest/images/ --evaluation_type our --dataset_name ALL
# iHarmony4 and HVIDIT datasets
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root  results/experiment/test_latest/images/ --evaluation_type our --dataset_name newALL
```
## Quantitative Result

<table class="tg">
  <tr>
    <th class="tg-0pky" align="center">Dataset</th>
    <th class="tg-0pky" align="center">Metrics</th>
    <th class="tg-0pky" align="center">Composite</th>
    <th class="tg-0pky" align="center">Ours<br>(iHarmony4)</th>
    <th class="tg-0pky" align="center">Ours<br>(iHarmony4+HVIDIT)</th>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HCOCO</td>
    <td class="tg-0pky" align="center">
        MSE</br>
        PSNR</br>
        SSIM</br>
        fMSE</br>
        fPSNR</br>
        fSSIM
    </td>
    <td class="tg-0pky" align="right">
        69.37</br>
        33.99</br>
        0.9853</br>
        996.59</br>
        19.86</br>
        0.8257
    </td>
    <td class="tg-0pky" align="right">
        21.61</br>
        37.82</br>
        0.9812</br>
        361.94</br>
        24.17</br>
        0.8736
    </td>
    <td class="tg-0pky" align="right">
        21.51</br>
        37.81</br>
        0.9812</br>        
        363.76</br>
        24.17</br>
        0.8735
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HAdobe5k</td>
    <td class="tg-0pky" align="center">
        MSE</br>
        PSNR</br>
        SSIM</br>
        fMSE</br>
        fPSNR</br>
        fSSIM
    </td>
    <td class="tg-0pky" align="right">
        345.54</br>
        28.52</br>
        0.9483</br>
        2051.61</br>
        17.52</br>
        0.7295
    </td>
    <td class="tg-0pky" align="right">
        40.67</br>
        36.61</br>
        0.9362</br>
        259.05</br>
        26.36</br>
        0.8413
    </td>
    <td class="tg-0pky" align="right">
        39.27</br>
        36.60</br>
        0.9364</br>
        259.91</br>
        26.32</br>
        0.8407
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HFlickr</td>
    <td class="tg-0pky" align="center">
        MSE</br>
        PSNR</br>
        SSIM</br>
        fMSE</br>
        fPSNR</br>
        fSSIM
    </td>
    <td class="tg-0pky" align="right">
        264.35</br>
        28.43</br>
        0.9620</br>
        1574.37</br>
        18.09</br>
        0.8036
    </td>
    <td class="tg-0pky" align="right">
        94.91</br>
        32.10</br>
        0.9614</br>
        638.36</br>
        21.97</br>
        0.8444
    </td>
    <td class="tg-0pky" align="right">
        94.25</br>
        32.06</br>
        0.9615</br>
        635.73</br>
        21.92</br>
        0.8436
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">Hday2night</td>
    <td class="tg-0pky" align="center">
        MSE</br>
        PSNR</br>
        SSIM</br>
        fMSE</br>
        fPSNR</br>
        fSSIM
    </td>
    <td class="tg-0pky" align="right">
        109.65</br>
        34.36</br>
        0.9607</br>
        1409.98</br>
        19.14</br>
        0.6353
    </td>
    <td class="tg-0pky" align="right">
        51.44</br>
        37.06</br>
        0.9308</br>
        740.59</br>
        22.40</br>
        0.6585
    </td>
    <td class="tg-0pky" align="right">
        59.87</br>
        36.42</br>
        0.9318</br>
        856.95</br>
        21.73</br>
        0.6549
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HVIDIT</td>
    <td class="tg-0pky" align="center">
        MSE</br>
        PSNR</br>
        SSIM</br>
        fMSE</br>
        fPSNR</br>
        fSSIM
    </td>
    <td class="tg-0pky" align="right">
        53.12</br>
        38.72</br>
        0.9922</br>        
        1604.41</br>
        19.01</br>
        0.7614
    </td>
    <td class="tg-0pky" align="right">
        -</br>
        -</br>
        -</br>
        -</br>
        -
    </td>
    <td class="tg-0pky" align="right">
        25.51</br>
        41.43</br>
        0.9919</br>
        738.66</br>
        21.86</br>
        0.7139
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">ALL</td>
    <td class="tg-0pky" align="center">
        MSE</br>
        PSNR</br>
        SSIM</br>
        fMSE</br>
        fPSNR</br>
        fSSIM
    </td>
    <td class="tg-0pky" align="right">
        167.39</br>
        32.07</br>
        0.9724</br>
        1386.12</br>
        18.97</br>
        0.7905
    </td>
    <td class="tg-0pky" align="right">
        35.90</br>
        36.81</br>
        0.9649</br>
        369.64</br>
        24.53</br>
        0.8571
    </td>
    <td class="tg-0pky" align="right">
        35.09</br>
        36.99</br>
        0.9662</br>
        388.30</br>
        24.39</br>
        0.8506
    </td>
  </tr>

</table>

## Real composite image harmonnization
More compared results can be found at [Google Drive](https://drive.google.com/file/d/10OIMil_whZ8HlJZobEnY6rZ1-a1I3F1i/view?usp=sharing) or [BaduCloud](https://pan.baidu.com/s/1UvKitGPXlszZH0PraFxswA) (access code: lgs2).

# Bibtex
If you use this code for your research, please cite our papers.


```
@InProceedings{Guo_2021_CVPR,
    author    = {Guo, Zonghui and Zheng, Haiyong and Jiang, Yufeng and Gu, Zhaorui and Zheng, Bing},
    title     = {Intrinsic Image Harmonization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16367-16376}
}
```

# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repo of [DoveNet](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4/tree/master/DoveNet) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 
