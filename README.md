## Progressive Feedback Enhanced Transformer for Image Forgery Localization  
An official implementation code for paper "Progressive Feedback Enhanced Transformer for Image Forgery Localization".
This repo will provide <B>codes, pretrained/trained weights, and our training datasets</B>. 

## Network Architecture
<center> <img src="fig/Network Architecture.png" alt="architecture"/> </center>

## Dependency
- torch 2.0.1
- torchvision 0.11.0
- scikit-learn 1.2.2

### Dataset
#### Summary 
The MBH−COCO dataset contains 30,000 spliced images and 15,000 copy-move images, generated using the MBH method. 
Additionally, the paper also mentions 5,000 inpainting images that are randomly selected from the public dataset [PSCC-Net](https://github.com/proteus1991/PSCC-Net), these are not included in the zip file.

The MBH−RAISE dataset contains 5,000 generated images, including both splicing and copy-move images.
The generation process of the datasets includes Matting, Alpha Blending, and Harmonization. The code is built on top of [SIM](https://github.com/nowsyn/SIM) and [colorRealism](https://github.com/jflalonde/colorRealism).

#### Dataset Download
You can download MBH-COCO and MBH-RAISE dataset on
[[Baiduyun Link]](https://pan.baidu.com/s/1L8E6695h47pzJ_yubLAJ2A) (extract code: vorh).

**As MBH-COCO dataset was created using MS COCO, you must follow the licensing terms of MS COCO.**

The generated MBH-RAISE dataset uses foreground images from the public matting datasets [AIM](https://github.com/JizhiziLi/AIM), [Distinctions](https://github.com/yuhaoliu7456/CVPR2020-HAttMatting), [PPM](https://github.com/Chinees-zhang/MODNet), [RWP](https://github.com/yucornetto/MGMatting), and [SIMD](https://github.com/nowsyn/SIM) dataset, combined with background images from the [RAISE](http://loki.disi.unitn.it/RAISE/) dataset.

**As MBH-RAISE dataset was created using the aforementioned datasets, you must follow relevant licenses.**
You are allowed to use the datasets for <B>research purpose only</B>.

## Training
- Prepare for the training dataset.
- Download the pre-trained model for backbone from: [[Google Drive Link]](https://drive.google.com/drive/folders/10NGg9hMN8AgUpcOpfAetTR37CFPAHGmT?usp=sharing) or [[Baiduyun Link]](https://pan.baidu.com/s/1FrNBKIX_tGzzQpDG83SBYQ) (extract code: 7f24).
- Modify configuration file. Set paths and settings properly in 'config.py'

```bash
python train.py 
```

## Testing
Download the weights from [Google Drive Link](https://drive.google.com/drive/folders/1aS0s7D3SweV9bGDepHgGXjgGHrfBtSg_?usp=sharing) or [Baiduyun Link](https://pan.baidu.com/s/15p-IX9yz82rm96k3BnefTQ) (extract code: `inc0`) and move it into the `checkpoint_save/`.

- **profact.pth** &emsp; MBH Datasets (50K+5K) for two-stage training.
- **profact_casia2.pth** &emsp;  Only CASIAv2 for one-stage training.
- **profact_ex.pth** &emsp; Publicly available datasets (in alignment with [TruFor](https://github.com/grip-unina/TruFor) and [CAT-Net](https://github.com/mjkwon2021/CAT-Net)), including CASIAv2, IMD, and FantasticReality, to extend the MBH Datasets for two-stage training.

For training models with only one stage, run:
```bash
python test_s1.py
```

For training models with two stages, run:
```bash
python test_s2.py
```
The predictions are saved in the `results` directory.


## License 
The code and dataset is released only for academic research. 
Commercial usage is strictly prohibited.

## Citation
 ```
@article{zhu2023progressive,
  title={Progressive Feedback-Enhanced Transformer for Image Forgery Localization},
  author={Zhu, Haochen and Cao, Gang and Huang, Xianglin},
  journal={arXiv preprint arXiv:2311.08910},
  year={2023}
}
```

## Contact
If you have any questions, please contact me(zhuhc_98@163.com).
