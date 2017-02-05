# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np

import io, os, sys, time, datetime
from PIL import Image
import pickle

import chainer
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F

from net import Encoder, Generator, Discriminator

parser = argparse.ArgumentParser(description='vae-gan for celeba')
parser.add_argument('--init_epoch', '-i', default=None,
                    help='Initialize the model from given file')
parser.add_argument('--output', '-o', default='test',
                    help='output file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--size', '-s', default=64, type=int, choices=[48, 64, 80, 96, 112, 128],
                    help='image size')
parser.add_argument("--data_dir", type=str, default="./data",
                    help='data directory')
parser.add_argument("--model_dir", type=str, default="./models",
                    help='model directory')
parser.add_argument("--visualization_dir", '-v', type=str, default="./images",
                    help='visualization directory')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
image_size = args.size

os.environ['PATH'] += ':/usr/local/cuda/bin'
print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))

try:
    os.mkdir(args.model_dir)
except:
    pass

try:
    os.mkdir(args.visualization_dir)
except:
    pass

sys.path.append('/home/mil/fukuta/datasets/celeba')
from celeba import CelebA

d = CelebA()
N = d.n_imgs
img_channel = 3
image_size = 64
attribute_size = 40

enc = Encoder(density=1, size=image_size)
gen = Generator(density=1, size=image_size)
dis = Discriminator(density=1, size=image_size)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    enc.to_gpu()
    gen.to_gpu()
    dis.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy

optimizer_enc = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_gen = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_dis = optimizers.Adam(alpha=0.0001, beta1=0.5)

optimizer_enc.setup(enc)
optimizer_gen.setup(gen)
optimizer_dis.setup(dis)

optimizer_enc.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

if args.init_epoch is not None:
    serializers.load_hdf5(args.input + '.enc.model', enc)
    serializers.load_hdf5(args.input + '.enc.state', optimizer_enc)
    serializers.load_hdf5(args.input + '.gen.model', gen)
    serializers.load_hdf5(args.input + '.gen.state', optimizer_gen)
    serializers.load_hdf5(args.input + '.dis.model', dis)
    serializers.load_hdf5(args.input + '.dis.state', optimizer_dis)


def visualize():
    # save original image
    x, _ = read_images(64, image_size, volatile=True)
    img_org = ((cuda.to_cpu(x.data) + 1) * 128).clip(0, 255).astype(np.uint8)

    # save reconstruction image
    mu, var = enc(x, train=False)
    # z = F.gaussian(mu, var)
    img_rec = gen(mu, train=False).data
    img_rec = ((cuda.to_cpu(img_rec) + 1) * 128).clip(0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for m in range(32):
        i = m / 8
        j = m % 8
        ax = fig.add_subplot(8, 8, 16 * i + j + 1, xticks=[], yticks=[])
        ax.imshow(img_org[m].transpose(1, 2, 0))
        ax = fig.add_subplot(8, 8, 16 * i + j + 8 + 1, xticks=[], yticks=[])
        ax.imshow(img_rec[m].transpose(1, 2, 0))
    plt.savefig('{}/reconstruction_{:03d}'.format(args.visualization_dir, epoch))
    plt.close()

    # save generated image
    z = Variable(xp.random.uniform(-1, 1, (64, 128)).astype(np.float32))
    img_gen = gen(z, train=False).data
    img_gen = ((cuda.to_cpu(img_gen) + 1) * 128).clip(0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(img_gen[i].transpose(1, 2, 0))
    fig.savefig('{}/generate_{:03d}'.format(args.visualization_dir, epoch))
    plt.close()


def read_images(batch_size, image_size, img_channel=3, volatile=False):
    x_batch = np.zeros((batch_size, img_channel, image_size, image_size), dtype=np.float32)
    attr_batch = np.zeros((batch_size, attribute_size), dtype=np.float32)
    for i in range(batch_size):
        data_index = np.random.randint(N)
        img = d.img_orig(data_index)
        attr = d.attributes[data_index]
        offset_x = np.random.randint(8) + 13
        offset_y = np.random.randint(8) + 33
        w = 144
        h = 144
        pixels = np.asarray(img.convert('RGB').crop((offset_x, offset_y, offset_x + w, offset_y + h)).resize(
            (image_size, image_size)))
        pixels = pixels.astype(np.float32).transpose((2, 0, 1)).reshape((3, image_size, image_size))
        x_batch[i] = pixels / 127.5 - 1
        attr_batch[i] = attr
    x_batch = Variable(xp.asarray(x_batch), volatile=volatile)
    attr_batch = Variable(xp.asarray(attr_batch), volatile=volatile)

    return x_batch, attr_batch


latent_size = gen.latent_size
out_image_num = 64
start_time = time.time()
train_count = 0
C = 1

for epoch in range(1, n_epoch + 1):
    print('Epoch {}'.format(epoch))

    sum_L_reconstruction = 0
    sum_L_enc = 0
    sum_L_gen = 0
    sum_L_dis = 0

    loop = 1000
    for i in range(loop):
        x, _ = read_images(batchsize, image_size)

        # encode
        mu_z, ln_var_z = enc(x)
        L_prior = F.gaussian_kl_divergence(mu_z, ln_var_z) / batchsize

        # generate from encoded z
        z0 = F.gaussian(mu_z, ln_var_z)
        x0 = gen(z0)
        y0, l0 = dis(x0)

        # generate from prior z
        z1 = Variable(xp.random.normal(0, 1, (batchsize, latent_size)).astype(np.float32))
        x1 = gen(z1)
        y1, l1 = dis(x1)

        # calculate gan loss
        L_gen0 = F.softmax_cross_entropy(y0, Variable(xp.zeros(batchsize).astype(np.int32)))
        L_gen1 = F.softmax_cross_entropy(y1, Variable(xp.zeros(batchsize).astype(np.int32)))

        L_dis0 = F.softmax_cross_entropy(y0, Variable(xp.ones(batchsize).astype(np.int32)))
        L_dis1 = F.softmax_cross_entropy(y1, Variable(xp.ones(batchsize).astype(np.int32)))

        # calculate reconstruction loss
        y2, l2 = dis(x)
        L_rec = F.mean_squared_error(l0, l2) * l0.data.shape[2] * l0.data.shape[3]
        # L_rec = F.gaussian_nll(l0, l2, Variable(xp.zeros(l0.data.shape).astype('float32'))) / batchsize
        L_dis2 = F.softmax_cross_entropy(y2, Variable(xp.zeros(batchsize).astype(np.int32)))

        L_enc = L_prior + L_rec
        L_gen = L_gen0 + L_gen1 + C * L_rec
        L_dis = L_dis0 + L_dis1 + L_dis2

        # update
        optimizer_enc.zero_grads()
        L_enc.backward()
        optimizer_enc.update()

        optimizer_gen.zero_grads()
        L_gen.backward()
        optimizer_gen.update()

        optimizer_dis.zero_grads()
        L_dis.backward()
        optimizer_dis.update()

        sum_L_enc += float(L_enc.data)
        sum_L_gen += float(L_gen.data)
        sum_L_dis += float(L_dis.data)

        sum_L_reconstruction += float(L_rec.data)

        train_count += 1
        duration = time.time() - start_time
        throughput = train_count * args.batchsize / duration
        sys.stderr.write(
            '\rtrain {} updates ({} samples) time: {} ({:.2f} images/sec)'
                .format(train_count, train_count * batchsize,
                        str(datetime.timedelta(seconds=duration)).split('.')[0], throughput))

        sys.stdout.flush()

    print()
    print(" reconstruction_loss={}".format(sum_L_reconstruction / loop))
    print(' enc loss={}, {}'.format(sum_L_enc / loop, sum_L_enc / loop - sum_L_reconstruction / loop))
    print(' gen loss={}, {}'.format(sum_L_gen / loop, sum_L_gen / loop - C * sum_L_reconstruction / loop))
    print(' dis loss={},'.format(sum_L_dis / loop))

    y0.to_cpu()
    p_gen_enc = y0.data.transpose(1, 0)
    p_gen_enc = np.exp(p_gen_enc)
    sum_p_gen_enc = p_gen_enc[0] + p_gen_enc[1]
    win_gen_enc = p_gen_enc[0] / sum_p_gen_enc
    print(" D(gen_enc)", win_gen_enc.mean())

    y1.to_cpu()
    p_gen_prior = y1.data.transpose(1, 0)
    p_gen_prior = np.exp(p_gen_prior)
    sum_p_gen_prior = p_gen_prior[0] + p_gen_prior[1]
    win_gen_prior = p_gen_prior[0] / sum_p_gen_prior
    print(" D(gen_prior) ", win_gen_prior.mean())

    y2.to_cpu()
    p_real = y2.data.transpose(1, 0)
    p_real = np.exp(p_real)
    sum_p_real = p_real[0] + p_real[1]
    win_real = p_real[0] / sum_p_real
    print(" D(real) ", win_real.mean())

    serializers.save_hdf5('{0}/{1}_{2:03d}.enc.model'.format(args.model_dir, args.output, epoch), enc)
    serializers.save_hdf5('{0}/{1}_{2:03d}.enc.state'.format(args.model_dir, args.output, epoch), optimizer_enc)
    serializers.save_hdf5('{0}/{1}_{2:03d}.gen.model'.format(args.model_dir, args.output, epoch), gen)
    serializers.save_hdf5('{0}/{1}_{2:03d}.gen.state'.format(args.model_dir, args.output, epoch), optimizer_gen)
    serializers.save_hdf5('{0}/{1}_{2:03d}.dis.model'.format(args.model_dir, args.output, epoch), dis)
    serializers.save_hdf5('{0}/{1}_{2:03d}.dis.state'.format(args.model_dir, args.output, epoch), optimizer_dis)

    visualize()
