## Implementation

This folder provides the PyTorch code and pretrained model of GhostNet on ImageNet.

`ghostnet.py` implemented `GhostModule` and `GhostNet`.

### Requirements
The code was verified on Python3.6, PyTorch-1.0+.

### Usage
Run `python validate.py --eval --data=/path/to/imagenet/dir/` to evaluate on `val` set.

You'll get the accuracy: top-1 acc=`0.7398` and top-5 acc=`0.9146` with only `142M` Flops (or say MAdds).

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

