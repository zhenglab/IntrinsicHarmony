<base target="_blank"/>


# Intrinsic Image Harmonization **[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Intrinsic_Image_Harmonization_CVPR_2021_paper.pdf)]**<br>
Zonghui Guo, Haiyong Zheng, Yufeng Jiang, Zhaorui Gu, Bing Zheng<br>


Here we provide PyTorch implementation and the trained model of our framework.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset, and our HVIDIT dataset [Google Drive](https://drive.google.com/file/d/1-pa_9BNgIkuR0j1gcCxh8GI3XSWZN0e7/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1DR600XJFhm8lqfHZ6mOU_A) (access code: akbi).

- Train a model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model retinexltifpm  --name retinexltifpm_allihd  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
- Test the model
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model retinexltifpm  --name retinexltifpm_allihd  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```

## Apply a pre-trained model
- Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1JyAzPDJkvYpeP6IpoD1kMuKMTWnOIDFu?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1xlVIBuTcfdPOsTRNQWTRsQ) (access code: 20m6), and put net_G.pth in the directory checkpoints/experiment. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model retinexltifpm  --name experiment  --dataset_root <dataset_dir> --dataset_name IHD --batch_size xx --init_port xxxx
```
## Evaluation
We provide the code in ih_evaluation.py. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root  results/experiment/test_latest/images/ --evaluation_type our --dataset_name ALL
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
        PSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        33.99</br>
        69.37</br>
        996.59
    </td>
    <td class="tg-0pky" align="right">
        37.61</br>
        23.25</br>
        386.39
    </td>
    <td class="tg-0pky" align="right">
        37.77</br>
        21.84</br>
        367.38
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HAdobe5k</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        28.52</br>
        345.54</br>
        2051.61
    </td>
    <td class="tg-0pky" align="right">
        36.20</br>
        42.21</br>
        296.76
    </td>
    <td class="tg-0pky" align="right">
        36.49</br>
        39.53</br>
        266.49
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HFlickr</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        28.43</br>
        264.35</br>
        1574.37
    </td>
    <td class="tg-0pky" align="right">
        31.74</br>
        100.86</br>
        676.71
    </td>
    <td class="tg-0pky" align="right">
        32.08</br>
        96.87</br>
        635.60
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">Hday2night</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        34.36</br>
        109.65</br>
        1409.98
    </td>
    <td class="tg-0pky" align="right">
        36.48</br>
        50.64</br>
        755.88
    </td>
    <td class="tg-0pky" align="right">
        36.60</br>
        50.37</br>
        763.33
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">HVIDIT</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        38.72</br>
        53.12</br>
        1604.41
    </td>
    <td class="tg-0pky" align="right">
        -</br>
        -</br>
        -
    </td>
    <td class="tg-0pky" align="right">
        41.83</br>
        22.49</br>
        691.06
    </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">ALL</td>
    <td class="tg-0pky" align="center">
        PSNR</br>
        MSE</br>
        fMSE
    </td>
    <td class="tg-0pky" align="right">
        32.07</br>
        167.39</br>
        1386.12
    </td>
    <td class="tg-0pky" align="right">
        36.53</br>
        37.95</br>
        399.34
    </td>
    <td class="tg-0pky" align="right">
        36.96</br>
        35.33</br>
        388.50
    </td>
  </tr>

</table>


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
