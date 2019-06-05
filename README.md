# SiamVGG

**SiamVGG: Visual Tracking with Deeper Siamese Networks**

The accuracy may be improved by using cv2 instead of PIL. I'll developing it later (if you are interested in it you can modified it yourself).

Won 4th place in [VOT2018 Realtime Challenge](http://www.votchallenge.net/vot2018/)

[Demo](https://www.youtube.com/watch?v=cvP64cmiAmY)

SiamVGG with RPN module [link](https://github.com/leeyeehoo/SiamRPN-VGG)

Required:

Python 2.7
PyTorch 4.0 (in VOTToolkit <=3.0)

* train: Including the training code.
* dataset: Instruction of generating the dataset note step by step.
* vot2018submission: Including the original submission in VOT2018. Download the pretrained data from [Google Drive](https://drive.google.com/file/d/13rx9kMJ1lwpics1Qr9_uKjloqLHfMaoU/view?usp=sharing) and follow the instruction to set up the test. (Partial of the code are modified from [SiamFC-pytorch](https://github.com/huanglianghua/siamfc-pytorch) and [https://github.com/HelloRicky123/Siamese-RPN], thanks to Lianghua and Ruiqi.)

If you find our project useful, please cite our technical report:
```
@inproceedings{Li2019SiamVGGVT,
  title={SiamVGG: Visual Tracking using Deeper Siamese Networks},
  author={Yuhong Li and Xiaofan Zhang},
  year={2019}
}
```
[arxiv version](https://arxiv.org/abs/1902.02804)
