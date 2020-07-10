## Implementation

This repo provides the TensorFlow code and pretrained model of GhostNet on ImageNet. The PyTorch implementation can be found at [https://github.com/iamhankai/ghostnet.pytorch](https://github.com/iamhankai/ghostnet.pytorch).

`myconv2d.py` implemented `GhostModule` and `ghostnet.py` implemented `GhostNet`.

### Requirements
The code was verified on Python3.6, TensorFlow-1.13.1, Tensorpack-0.9.7. Not sure on other version.

### Usage
Run `python main.py --eval --data_dir=/path/to/imagenet/dir/ --load=./models/ghostnet_checkpoint` to evaluate on `val` set.

You'll get the accuracy: top-1 error=`0.26066`, top-5 error=`0.08614` with only `141M` Flops (or say MAdds).

### Data Preparation
ImageNet data dir should have the following structure, and `val` and `caffe_ilsvrc12` subdirs are essential:
```
dir/
  train/
    ...
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
      ...
    ...
  caffe_ilsvrc12/
    ...
```
caffe_ilsvrc12 data can be downloaded from http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

