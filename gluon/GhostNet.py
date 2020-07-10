#it's from https://github.com/osmr/imgclsmob.git
import os
import math
from mxnet import cpu
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from inspect import isfunction

class ReLU6(HybridBlock):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0.0, 6.0, name="relu6")


class PReLU2(HybridBlock):
    """
    Parametric leaky version of a Rectified Linear Unit (with wide alpha).
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    alpha_initializer : Initializer
        Initializer for the `embeddings` matrix.
    """
    def __init__(self,
                 in_channels=1,
                 alpha_initializer=mx.init.Constant(0.25),
                 **kwargs):
        super(PReLU2, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get("alpha", shape=(in_channels,), init=alpha_initializer)

    def hybrid_forward(self, F, x, alpha):
        return F.LeakyReLU(x, gamma=alpha, act_type="prelu", name="fwd")


class HSigmoid(HybridBlock):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0


class HSwish(HybridBlock):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x * F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0

def get_activation_layer(activation):
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu6":
            return ReLU6()
        elif activation == "swish":
            return nn.Swish()
        elif activation == "hswish":
            return HSwish()
        elif activation == "hsigmoid":
            return HSigmoid()
        else:
            return nn.Activation(activation)
    else:
        assert (isinstance(activation, HybridBlock))
        return activation

def round_channels(channels,
                   divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def conv1x1(in_channels,
            out_channels,
            strides=1,
            groups=1,
            use_bias=False):
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        in_channels=in_channels)


def conv3x3(in_channels,
            out_channels,
            strides=1,
            padding=1,
            dilation=1,
            groups=1,
            use_bias=False):
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        in_channels=in_channels)


class ConvBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=in_channels)
            if self.use_bn:
                self.bn = nn.BatchNorm(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x



def conv1x1_block(in_channels,
                  out_channels,
                  strides=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


def conv3x3_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                 groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
    
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)

def dwconv_block(in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)



def dwconv3x3_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=1,
                    dilation=1,
                    use_bias=False,
                    bn_epsilon=1e-5,
                    bn_use_global_stats=False,
                    activation=(lambda: nn.Activation("relu")),
                    **kwargs):
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)

def dwconv5x5_block(in_channels,
                    out_channels,
                    strides=1,
                    padding=2,
                    dilation=1,
                    use_bias=False,
                    bn_epsilon=1e-5,
                    bn_use_global_stats=False,
                    activation=(lambda: nn.Activation("relu")),
                    **kwargs):
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)

class DwsConvBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 dw_activation=(lambda: nn.Activation("relu")),
                 pw_activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.dw_conv = dwconv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                activation=dw_activation)
            self.pw_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                activation=pw_activation)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

def dwsconv3x3_block(in_channels,
                     out_channels,
                     strides=1,
                     padding=1,
                     dilation=1,
                     use_bias=False,
                     bn_epsilon=1e-5,
                     bn_use_global_stats=False,
                     dw_activation=(lambda: nn.Activation("relu")),
                     pw_activation=(lambda: nn.Activation("relu")),
                     **kwargs):
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        **kwargs)

class SEBlock(HybridBlock):
    def __init__(self,
                 channels,
                 reduction=16,
                 round_mid=False,
                 mid_activation=(lambda: nn.Activation("relu")),
                 out_activation=(lambda: nn.Activation("sigmoid")),
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                use_bias=True)
            self.activ = get_activation_layer(mid_activation)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels,
                use_bias=True)
            self.sigmoid = get_activation_layer(out_activation)
    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = F.broadcast_mul(x, w)
        return x


class GhostHSigmoid(HybridBlock):
    """
    Approximated sigmoid function, specific for GhostNet.
    """
    def __init__(self, **kwargs):
        super(GhostHSigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0.0, 1.0)

class GhostConvBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(GhostConvBlock, self).__init__(**kwargs)
        main_out_channels = math.ceil(0.5 * out_channels)
        cheap_out_channels = out_channels - main_out_channels

        with self.name_scope():
            self.main_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=main_out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation)
            self.cheap_conv = dwconv3x3_block(
                in_channels=main_out_channels,
                out_channels=cheap_out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation)

    def hybrid_forward(self, F, x):
        x = self.main_conv(x)
        y = self.cheap_conv(x)
        return F.concat(x, y, dim=1)  

class GhostExpBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_kernel3,
                 exp_factor,
                 use_se,
                 bn_use_global_stats=False,
                 **kwargs):
        super(GhostExpBlock, self).__init__(**kwargs)
        self.use_dw_conv = (strides != 1)
        self.use_se = use_se
        mid_channels = int(math.ceil(exp_factor * in_channels))

        with self.name_scope():
            self.exp_conv = GhostConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            if self.use_dw_conv:
                dw_conv_class = dwconv3x3_block if use_kernel3 else dwconv5x5_block
                self.dw_conv = dw_conv_class(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            if self.use_se:
                self.se = SEBlock(
                    channels=mid_channels,
                    reduction=4,
                    out_activation=GhostHSigmoid())
            self.pw_conv = GhostConvBlock(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.exp_conv(x)
        if self.use_dw_conv:
            x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x

class GhostUnit(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_kernel3,
                 exp_factor,
                 use_se,
                 bn_use_global_stats=False,
                 **kwargs):
        super(GhostUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = GhostExpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                use_kernel3=use_kernel3,
                exp_factor=exp_factor,
                use_se=use_se,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    pw_activation=None)

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class GhostClassifier(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 **kwargs):
        super(GhostClassifier, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GhostNet(HybridBlock):
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 classifier_mid_channels,
                 kernels3,
                 exp_factors,
                 use_se,
                 first_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(GhostNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                        use_kernel3 = kernels3[i][j] == 1
                        exp_factor = exp_factors[i][j]
                        use_se_flag = use_se[i][j] == 1
                        stage.add(GhostUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            use_kernel3=use_kernel3,
                            exp_factor=exp_factor,
                            use_se=use_se_flag,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=final_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = final_block_channels
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(GhostClassifier(
                in_channels=in_channels,
                out_channels=classes,
                mid_channels=classifier_mid_channels))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_ghostnet(width_scale=1.0,
                 model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    init_block_channels = 16
    channels = [[16], [24, 24], [40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160, 160, 160]]
    kernels3 = [[1], [1, 1], [0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    exp_factors = [[1], [3, 3], [3, 3], [6, 2.5, 2.3, 2.3, 6, 6], [6, 6, 6, 6, 6]]
    use_se = [[0], [0, 0], [1, 1], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1]]
    final_block_channels = 960
    classifier_mid_channels = 1280
    first_stride = False

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale, divisor=4) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale, divisor=4)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale, divisor=4)

    net = GhostNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        use_se=use_se,
        first_stride=first_stride,
        **kwargs)

    

    return net


def ghostnet(**kwargs):
    return get_ghostnet(model_name="ghostnet", **kwargs)

def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        ghostnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ghostnet or weight_count == 5180840)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
