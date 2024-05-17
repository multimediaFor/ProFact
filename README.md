## Progressive Feedback Enhanced Transformer for Image Forgery Localization  
An official implementation code for paper "Progressive Feedback Enhanced Transformer for Image Forgery Localization".
This repo will provide <B>codes, pretrained/trained weights, and our training datasets</B>. 

 
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

### License 
The code and datasets are released only for academic research. 
Commercial usage is strictly prohibited.

### Citation
 ```
@article{zhu2023progressive,
  title={Progressive Feedback-Enhanced Transformer for Image Forgery Localization},
  author={Zhu, Haochen and Cao, Gang and Huang, Xianglin},
  journal={arXiv preprint arXiv:2311.08910},
  year={2023}
}
```
