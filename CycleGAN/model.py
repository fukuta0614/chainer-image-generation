import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, ksize=4, stride=2, pad=1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample == 'down':
            layers['c'] = L.Convolution2D(ch0, ch1, ksize, stride, pad, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, ksize, stride, pad, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x, train):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h, test=not train)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


class ResBlock(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['conv1'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w, nobias=True)
        layers['conv2'] = L.Convolution2D(ch1, ch1, 3, 1, 1, initialW=w, nobias=True)
        if bn:
            layers['bn1'] = L.BatchNormalization(ch1)
            layers['bn2'] = L.BatchNormalization(ch1)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = self.bn2(self.conv2(h), test=not train)

        # return F.relu(h + x)
        return h + x


class Generator(chainer.Chain):
    def __init__(self, ch=128, block_num=9, bn=True):
        layers = {}
        self.block_num = block_num
        layers['conv1'] = CBR(3, ch // 4, ksize=5, stride=1, pad=2, bn=bn, sample='down')
        layers['conv2'] = CBR(ch // 4, ch // 2, ksize=3, stride=2, pad=1, bn=bn, sample='down')
        layers['conv3'] = CBR(ch // 2, ch, ksize=3, stride=2, pad=1, bn=bn, sample='down')
        for i in range(self.block_num):
            layers['res{}'.format(i)] = ResBlock(ch, ch, bn=bn)
        layers['dc1'] = CBR(ch, ch // 2, ksize=4, stride=2, pad=1, bn=bn, sample='up')
        layers['dc2'] = CBR(ch // 2, ch // 4, ksize=4, stride=2, pad=1, bn=bn, sample='up')
        layers['dc3'] = CBR(ch // 4, 3, ksize=5, stride=1, pad=2, bn=bn, sample='down')
        super(Generator, self).__init__(**layers)

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def __call__(self, x, train=True):
        h = self.conv1(x, train)
        h = self.conv2(h, train)
        h = self.conv3(h, train)
        for i in range(self.block_num):
            h = getattr(self, 'res{}'.format(i))(h, train)
        h = self.dc1(h, train)
        h = self.dc2(h, train)
        h = self.dc3(h, train)
        return F.tanh(h)


class Discriminator(chainer.Chain):
    def __init__(self):
        # Patch GAN
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(3, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = F.Convolution2D(512, 1, 3, 1, 1, initialW=w)
        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, train=True):
        h = self.c0_0(x_0, train=train)
        h = self.c1(h, train=train)
        h = self.c2(h, train=train)
        h = self.c3(h, train=train)
        h = self.c4(h)
        h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h


class GeneratorOld(chainer.Chain):
    def __init__(self, ch=128, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            conv1=L.Convolution2D(3, ch // 4, 5, 1, 2, initialW=w),
            conv2=L.Convolution2D(ch // 4, ch // 2, 3, 2, 1, initialW=w),
            conv3=L.Convolution2D(ch // 2, ch, 3, 2, 1, initialW=w),
            res1=ResBlock(ch, ch, initialW=w),
            res2=ResBlock(ch, ch, initialW=w),
            res3=ResBlock(ch, ch, initialW=w),
            res4=ResBlock(ch, ch, initialW=w),
            res5=ResBlock(ch, ch, initialW=w),
            res6=ResBlock(ch, ch, initialW=w),
            res7=ResBlock(ch, ch, initialW=w),
            res8=ResBlock(ch, ch, initialW=w),
            res9=ResBlock(ch, ch, initialW=w),
            dc1=L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w),
            dc2=L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w),
            dc3=L.Convolution2D(ch // 4, 3, 5, 1, 2, initialW=w),
        )

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def __call__(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.res1(h, train)
        h = self.res2(h, train)
        h = self.res3(h, train)
        h = self.res4(h, train)
        h = self.res5(h, train)
        h = self.res6(h, train)
        h = self.res7(h, train)
        h = self.res8(h, train)
        h = self.res9(h, train)
        h = F.relu(self.dc1(h))
        h = F.relu(self.dc2(h))
        h = self.dc3(h)

        return F.tanh(h)
