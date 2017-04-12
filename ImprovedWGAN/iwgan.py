import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self, n_hidden, activate='sigmoid', size=64, ch=512, wscale=0.02):
        assert (size % 8 == 0)
        initial_size = size // 8
        self.n_hidden = n_hidden
        if activate == 'sigmoid':
            self.activate = F.sigmoid
        elif activate == 'tanh':
            self.activate = F.tanh
        else:
            raise ValueError('invalid activate function')
        self.ch = ch
        self.initial_size = initial_size
        w = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0=L.Linear(self.n_hidden, initial_size * initial_size * ch, initialW=w),
            dc1=L.Deconvolution2D(ch // 1, ch // 2, 4, 2, 1, initialW=w),
            dc2=L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w),
            dc3=L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w),
            dc4=L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w),
        )

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(numpy.float32)

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def __call__(self, z, train=True):
        h = F.reshape(F.relu(self.l0(z)), (z.data.shape[0], self.ch, self.initial_size, self.initial_size))
        h = F.relu(self.dc1(h))
        h = F.relu(self.dc2(h))
        h = F.relu(self.dc3(h))
        x = self.activate(self.dc4(h))
        return x


class Discriminator(chainer.Chain):
    def __init__(self, size=64, ch=512, wscale=0.005):
        assert (size % 8 == 0)
        initial_size = size // 8

        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0_0=L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w),
            c0_1=L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w),
            c1_0=L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w),
            c1_1=L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w),
            c2_0=L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w),
            c2_1=L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w),
            c3_0=L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w),
            l4=L.Linear(initial_size * initial_size * ch, 1, initialW=w),
        )

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def __call__(self, x, train=True):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.c0_1(h))
        h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c1_1(h))
        h = F.leaky_relu(self.c2_0(h))
        h = F.leaky_relu(self.c2_1(h))
        h = F.leaky_relu(self.c3_0(h))
        h = self.l4(h)
        return F.sum(h) / h.size
