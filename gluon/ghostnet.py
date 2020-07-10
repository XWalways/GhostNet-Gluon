#it's modified from official pytorch version
import mxnet
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as F
import math
from mxnet.gluon.nn import HybridBlock

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class HardSigmoid(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        #self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return F.clip(x + 3, 0, 6, name="hard_sigmoid") / 6.

class SqueezeExcite(HybridBlock):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.Activation('relu'), gate_fn=HardSigmoid(), divisor=4):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.GlobalAvgPool2D()

        self.conv_reduce = nn.Conv2D(reduced_chs, kernel_size=1, in_channels=in_chs, use_bias=True)
        self.act1 = act_layer
        self.conv_expand = nn.Conv2D(in_chs, kernel_size=1, in_channels=reduced_chs, use_bias=True)

    def hybrid_forward(self, F, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = F.broadcast_mul(self.gate_fn(x_se), x)
        return x 

class ConvBnAct(HybridBlock):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.Activation('relu')):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2D(out_chs, in_channels=in_chs, kernel_size=kernel_size, strides=stride, padding=kernel_size//2, use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=out_chs, momentum=0.1)
        self.act1 = act_layer

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModule(HybridBlock):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        with self.name_scope():
            self.primary_conv = nn.HybridSequential(prefix='primary_conv_')
            self.primary_conv.add(nn.Conv2D(init_channels, in_channels=inp, kernel_size=kernel_size, strides=stride, padding=kernel_size//2, use_bias=False))
            self.primary_conv.add(nn.BatchNorm(in_channels=init_channels, momentum=0.1))
            if relu:
                self.primary_conv.add(nn.Activation('relu'))
            else:
                self.primary_conv.add(nn.HybridSequential())
       
            self.cheap_operation = nn.HybridSequential(prefix='cheap_operation_')
            self.cheap_operation.add(nn.Conv2D(new_channels, in_channels=init_channels, kernel_size=dw_size, strides=1, padding=dw_size//2, 
                                               groups=init_channels, use_bias=False))
            self.cheap_operation.add(nn.BatchNorm(in_channels=new_channels, momentum=0.1))
            if relu:
                self.cheap_operation.add(nn.Activation('relu'))
            else:
                self.cheap_operation.add(nn.HybridSequential())
        
    def hybrid_forward(self, F, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = F.concat(x1, x2, dim=1)
        return F.slice(out, begin=(None, 0, None, None), end=(None, self.oup, None, None))


class GhostBottleneck(HybridBlock):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.Activation('relu'), se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2D(mid_chs, in_channels=mid_chs, kernel_size=dw_kernel_size, strides=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, use_bias=False)
            self.bn_dw = nn.BatchNorm(in_channels=mid_chs, momentum=0.1)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.HybridSequential()
        else:
            with self.name_scope():
                self.shortcut = nn.HybridSequential()
                self.shortcut.add(nn.Conv2D(in_chs, in_channels=in_chs, kernel_size=dw_kernel_size, strides=stride,
                                            padding=(dw_kernel_size-1)//2, groups=in_chs, use_bias=False))
                self.shortcut.add(nn.BatchNorm(in_channels=in_chs, momentum=0.1))
                self.shortcut.add(nn.Conv2D(out_chs, in_channels=in_chs, kernel_size=1, strides=1, padding=0, use_bias=False))
                self.shortcut.add(nn.BatchNorm(in_channels=out_chs, momentum=0.1))
            
    def hybrid_forward(self, F, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x = x + self.shortcut(residual)
        return x


class GhostNet(HybridBlock):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2D(output_channel, in_channels=3, kernel_size=3, strides=2, padding=1, use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=output_channel, momentum=0.1)
        self.act1 = nn.Activation('relu')
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            with self.name_scope():
                stage = nn.HybridSequential()
            for i in range(len(layers)):    
                stage.add(layers[i])
            stages.append(stage)

        output_channel = _make_divisible(exp_size * width, 4)
        with self.name_scope():
            convbnrelu = nn.HybridSequential()
            convbnrelu.add(ConvBnAct(input_channel, output_channel, 1))
        stages.append(convbnrelu)
        input_channel = output_channel
        
        with self.name_scope():
            self.blocks = nn.HybridSequential()
            for i in range(len(stages)):
                self.blocks.add(stages[i])        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.GlobalAvgPool2D()
        self.conv_head = nn.Conv2D(output_channel, in_channels=input_channel, kernel_size=1, strides=1, padding=0, use_bias=True)
        self.act2 = nn.Activation('relu')
        self.classifier = nn.Dense(num_classes, in_units=output_channel)

    def hybrid_forward(self, F, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        
        
        if self.dropout > 0.:
            x = F.Dropout(x, p=self.dropout, mode='training')
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__=='__main__':
    model = ghostnet()
    model.hybridize()
    model.initialize()
    #input = mxnet.nd.random.uniform(-1, 1, shape=(32, 3, 320, 256))
    print(model)
    input = mxnet.nd.random.randn(32,3,224,224)
    y = model(input)
    print(y.shape)
