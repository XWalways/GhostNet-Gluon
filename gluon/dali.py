# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator
import horovod.mxnet as hvd
import os
import math
from types import SimpleNamespace
import mxnet as mx

class HybridTrainPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, prefetch_queue_depth = prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=args.shuffle, shard_id=shard_id, num_shards=num_shards)

        if dali_cpu:
            dali_device = "cpu"
            if args.dali_fuse_decoder:
                self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB)
            else:
                self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            if args.dali_fuse_decoder:
                # self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB,
                self.decode = ops.ImageDecoderRandomCrop(device="mixed", output_type=types.RGB,
                                                          random_area=args.random_area, random_aspect_ratio=args.random_aspect_ratio,
                                                          device_memory_padding=nvjpeg_padding, host_memory_padding=nvjpeg_padding)
            else:
                # self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
                self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                                device_memory_padding=nvjpeg_padding, host_memory_padding=nvjpeg_padding)

        if args.dali_fuse_decoder:
            self.resize = ops.Resize(device=dali_device, resize_x=crop_shape[1], resize_y=crop_shape[0])
        else:
            self.resize = ops.RandomResizedCrop(device=dali_device, size=crop_shape, random_area=args.random_area,
                                                random_aspect_ratio=args.random_aspect_ratio)

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
                                            image_type=types.RGB, mean=args.rgb_mean, std=args.rgb_std)
        if args.random_mirror:
            self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, prefetch_queue_depth=prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        if dali_cpu:
            dali_device = "cpu"
            # self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                            device_memory_padding=nvjpeg_padding,
                                            host_memory_padding=nvjpeg_padding)
        self.resize = ops.Resize(device=dali_device, resize_shorter=resize_shp) if resize_shp else None
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
                                            image_type=types.RGB, mean=args.rgb_mean, std=args.rgb_std)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]


def get_rec_iter(args, train, kv=None, dali_cpu=False):
    gpus = args.gpus
    num_threads = args.num_threads
    pad_output = (args.image_shape[0] == 4)

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC if args.input_layout == 'NHWC' else types.NCHW

    if 'horovod' in args.kv_store:
        rank = hvd.rank()
        nWrk = hvd.size()
    else:
        rank = kv.rank if kv else 0
        nWrk = kv.num_workers if kv else 1

    batch_size = args.batch_size // nWrk // len(gpus)

    if train:
        pipes = [HybridTrainPipe(args           = args,
                                 batch_size     = batch_size,
                                 num_threads    = num_threads,
                                 device_id      = gpu_id,
                                 rec_path       = args.rec_file,
                                 idx_path       = args.rec_file_idx,
                                 shard_id       = gpus.index(gpu_id) + len(gpus)*rank,
                                 num_shards     = len(gpus)*nWrk,
                                 crop_shape     = args.image_shape[1:],
                                 output_layout  = output_layout,
                                 dtype          = args.dtype,
                                 pad_output     = pad_output,
                                 dali_cpu       = dali_cpu,
                                 nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                 prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus]
    else:
        pipes = [HybridValPipe(  args           = args,
                                 batch_size     = batch_size,
                                 num_threads    = 1,
                                 device_id      = gpu_id,
                                 rec_path       = args.rec_file,
                                 idx_path       = args.rec_file_idx,
                                 shard_id       = 0 if args.dali_separ_val
                                                      else gpus.index(gpu_id) + len(gpus)*rank,
                                 num_shards     = 1 if args.dali_separ_val else len(gpus)*nWrk,
                                 crop_shape     = args.image_shape[1:],
                                 resize_shp     = args.data_val_resize,
                                 output_layout  = output_layout,
                                 dtype          = args.dtype,
                                 pad_output     = pad_output,
                                 dali_cpu       = dali_cpu,
                                 nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                 prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus] if args.data_val else None
    pipes[0].build()
    if not train:
        worker_val_examples = pipes[0].epoch_size("Reader")
        if not args.dali_separ_val:
            worker_val_examples = worker_val_examples // nWrk
            if rank < pipes[0].epoch_size("Reader") % nWrk:
                worker_val_examples += 1
        dali_iter = DALIClassificationIterator(pipes, worker_val_examples,
                                                   fill_last_batch=False) if args.data_val else None
    else:
        if args.num_examples < pipes[0].epoch_size("Reader"):
            warnings.warn("{} training examples will be used, although full training set contains {} examples".format(
                args.num_examples, pipes[0].epoch_size("Reader")))
        dali_iter = DALIClassificationIterator(pipes, args.num_examples // nWrk)

    return dali_iter



def get_data_rec(image_shape, crop_ratio, rec_file, rec_file_idx,
                 batch_size, num_workers, train=True, shuffle=True,
                 backend='dali-gpu', gpu_ids=[], kv_store='nccl', dtype='float16',
                 input_layout='NCHW'):

    args = SimpleNamespace()
    args.rec_file = os.path.expanduser(rec_file)
    args.rec_file_idx = os.path.expanduser(rec_file_idx)
    args.shuffle = shuffle
    args.image_shape = image_shape
    args.num_threads = num_workers

    args.random_mirror = True
    args.data_val  = True
    args.jitter_param = 0.4
    args.lighting_param = 0.1
    args.random_area = [0.08, 1.0]
    args.random_aspect_ratio = [3./4, 4./3]
    args.rgb_std = [58.393, 57.12, 57.375]
    args.rgb_mean = [123.68, 116.779, 103.939]
    args.data_val_resize = int(math.ceil(image_shape[-1] / crop_ratio))

    if backend == 'dali-gpu' or backend == 'dali-cpu':
        import mxnet as mx
        assert gpu_ids, ValueError('gpu_ids should not be empty')
        assert kv_store, ValueError('kv_store should not be empty')
        args.gpus = gpu_ids
        args.dali_validation_threads = 1
        args.input_layout = input_layout
        args.kv_store = kv_store
        args.dtype = dtype
        args.batch_size = batch_size * len(gpu_ids)
        args.dali_separ_val = False
        args.num_examples = 1281167
        args.dali_nvjpeg_memory_padding = 64
        args.dali_prefetch_queue = 3
        args.dali_fuse_decoder = 1
        args.dali_cpu = True if backend == 'dali-cpu' else False
        kv = None if 'horovod' in kv_store else mx.kvstore.create(kv_store)

        data_loader = get_rec_iter(train=train, args=args, kv=kv)
    else:
        raise NotImplementedError('data backend {} is not implemented.'.format(backend))

    return data_loader
