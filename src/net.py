import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable


class Encoder(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=128, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Encoder, self).__init__(
            enc1=L.Convolution2D(channel, 64 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            norm1=L.BatchNormalization(64 * density),
            enc2=L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm2=L.BatchNormalization(128 * density),
            enc3=L.Convolution2D(128 * density, 128 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm3=L.BatchNormalization(128 * density),
            enc4=L.Convolution2D(128 * density, 256 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm4=L.BatchNormalization(256 * density),
            mean=L.Linear(initial_size * initial_size * 256 * density, latent_size,
                          wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
            ln_var=L.Linear(initial_size * initial_size * 256 * density, latent_size,
                            wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
        )

    def __call__(self, x, train=True):
        h1 = F.relu(self.norm1(self.enc1(x), test=not train))
        h2 = F.relu(self.norm2(self.enc2(h1), test=not train))
        h3 = F.relu(self.norm3(self.enc3(h2), test=not train))
        h4 = F.relu(self.norm4(self.enc4(h3), test=not train))
        mean = self.mean(h4)
        ln_var = self.ln_var(h4)

        return mean, ln_var


class Generator(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=128, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Generator, self).__init__(
            g1=L.Linear(latent_size, initial_size * initial_size * 256 * density, wscale=0.02 * math.sqrt(latent_size)),
            norm1=L.BatchNormalization(initial_size * initial_size * 256 * density),
            g2=L.Deconvolution2D(256 * density, 128 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 256 * density)),
            norm2=L.BatchNormalization(128 * density),
            g3=L.Deconvolution2D(128 * density, 64 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm3=L.BatchNormalization(64 * density),
            g4=L.Deconvolution2D(64 * density, 32 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm4=L.BatchNormalization(32 * density),
            g5=L.Deconvolution2D(32 * density, channel, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)),
                       (z.data.shape[0], 256 * self.density, self.initial_size, self.initial_size))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        h5 = F.tanh(self.g5(h4))
        return h5


class Discriminator(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Discriminator, self).__init__(
            dis1=L.Convolution2D(channel, 32 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            dis2=L.Convolution2D(32 * density, 64 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm2=L.BatchNormalization(64 * density),
            dis3=L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm3=L.BatchNormalization(128 * density),
            dis4=L.Convolution2D(128 * density, 256 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm4=L.BatchNormalization(256 * density),
            dis5=L.Linear(initial_size * initial_size * 256 * density, 512,
                          wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
            norm5=L.BatchNormalization(512),
            dis6=L.Linear(512, 2, wscale=0.02 * math.sqrt(512)),
        )

    def __call__(self, x, train=True):
        h1 = F.relu(self.dis1(x))
        h2 = F.relu(self.norm2(self.dis2(h1), test=not train))
        h3 = F.relu(self.norm3(self.dis3(h2), test=not train))
        h4 = F.relu(self.norm4(self.dis4(h3), test=not train))
        h5 = F.relu(self.norm5(self.dis5(h4), test=not train))
        h6 = self.dis6(h5)
        return h6, h2


class Encoder_origin(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Encoder_origin, self).__init__(
            enc1=L.Convolution2D(channel, 32 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            enc2=L.Convolution2D(32 * density, 64 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm2=L.BatchNormalization(64 * density),
            enc3=L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm3=L.BatchNormalization(128 * density),
            enc4=L.Convolution2D(128 * density, 256 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm4=L.BatchNormalization(256 * density),
            mean=L.Linear(initial_size * initial_size * 256 * density, latent_size,
                          wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
            ln_var=L.Linear(initial_size * initial_size * 256 * density, latent_size,
                            wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.enc1(x))
        h2 = F.leaky_relu(self.norm2(self.enc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.enc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.enc4(h3), test=not train))
        mean = self.mean(h4)
        ln_var = self.ln_var(h4)

        return mean, ln_var


class Generator_origin(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Generator_origin, self).__init__(
            g1=L.Linear(latent_size, initial_size * initial_size * 256 * density, wscale=0.02 * math.sqrt(latent_size)),
            norm1=L.BatchNormalization(initial_size * initial_size * 256 * density),
            g2=L.Deconvolution2D(256 * density, 128 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 256 * density)),
            norm2=L.BatchNormalization(128 * density),
            g3=L.Deconvolution2D(128 * density, 64 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm3=L.BatchNormalization(64 * density),
            g4=L.Deconvolution2D(64 * density, 32 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm4=L.BatchNormalization(32 * density),
            g5=L.Deconvolution2D(32 * density, channel, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)),
                       (z.data.shape[0], 256 * self.density, self.initial_size, self.initial_size))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.tanh(self.g5(h4))


class Discriminator_org(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Discriminator_org, self).__init__(
            dis1=L.Convolution2D(channel, 32 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            dis2=L.Convolution2D(32 * density, 64 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm2=L.BatchNormalization(64 * density),
            dis3=L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm3=L.BatchNormalization(128 * density),
            dis4=L.Convolution2D(128 * density, 256 * density, 4, stride=2, pad=1,
                                 wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm4=L.BatchNormalization(256 * density),
            dis5=L.Linear(initial_size * initial_size * 256 * density, 2,
                          wscale=0.02 * math.sqrt(initial_size * initial_size * 256 * density)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.dis1(x))
        h2 = F.leaky_relu(self.norm2(self.dis2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dis3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dis4(h3), test=not train))
        return self.dis5(h4), h3
