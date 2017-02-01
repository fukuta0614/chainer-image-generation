# -*- coding: utf-8 -*-
import os, re, math, pylab
from math import *
import numpy as np

import matplotlib.patches as mpatches

import chainer

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


def read_images(batch_size, image_size, img_channel=3, volatile=False):
    x_batch = np.zeros((batch_size, img_channel, image_size, image_size), dtype=np.float32)
    attr_batch = np.zeros((batch_size, attribute_size), dtype=np.float32)
    for i in range(batch_size):
        data_index = np.random.randint(N)
        img = d.img_orig(data_index)
        attr = d.attributes(data_index)
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


def visualize():
    # save original image
    x, _ = read_images(64, image_size, volatile=True)
    img_org = ((cuda.to_cpu(x.data) + 1) * 128).clip(0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(9, 18))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
    for i in xrange(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(img_org[i].transpose(1, 2, 0))
    plt.savefig('{}/original_{:03d}'.format(args.validation_dir, epoch))

    # save reconstruction image
    mu, var = enc(x, train=False)
    img_rec = gen(mu, train=False).data
    img_rec = ((cuda.to_cpu(img_rec) + 1) * 128).clip(0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(9, 18))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(img_rec[i].transpose(1, 2, 0))
    plt.savefig('{}/reconstruction_{:03d}'.format(args.validation_dir, epoch))

    # save generated image
    z = Variable(xp.random.uniform(-1, 1, (64, latent_size)).astype(np.float32))
    img_gen = gen(z, train=False).data
    img_gen = ((cuda.to_cpu(img_gen) + 1) * 128).clip(0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(9, 18))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(img_gen[i].transpose(1, 2, 0))
    plt.savefig('{}/generate_{:03d}'.format(args.validation_dir, epoch))

