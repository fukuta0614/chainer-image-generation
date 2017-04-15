import argparse
import os, sys
import numpy as np
import datetime
import time
import pickle
import random
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logger
from model import Generator, Discriminator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ImageDataset


def progress_report(count, start_time, batchsize):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r{} updates ({} samples) time: {} ({:.2f} samples/sec)'.format(
            count, count * batchsize, str(datetime.timedelta(seconds=duration)).split('.')[0], throughput
        )
    )


def visualize(genA, genB, realA, realB, epoch, savedir):
    img_realA = ((realA + 1) * 127.5).clip(0, 255).astype(np.uint8)
    x_fakeB = genB(chainer.Variable(genB.xp.asarray(realA, 'float32'), volatile=True), train=False)
    x_recA = genA(x_fakeB, train=False)
    img_fakeB = ((cuda.to_cpu(x_fakeB.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img_recA = ((cuda.to_cpu(x_recA.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    img_realB = ((realB + 1) * 127.5).clip(0, 255).astype(np.uint8)
    x_fakeA = genA(chainer.Variable(genA.xp.asarray(realB, 'float32'), volatile=True), train=False)
    x_recB = genB(x_fakeA, train=False)
    img_fakeA = ((cuda.to_cpu(x_fakeA.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img_recB = ((cuda.to_cpu(x_recB.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(15, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(60):
        ax = fig.add_subplot(6, 10, i + 1, xticks=[], yticks=[])
        if i < 10:
            ax.imshow(img_realA[i].transpose(1, 2, 0))
        elif i < 20:
            ax.imshow(img_fakeB[i - 10].transpose(1, 2, 0))
        elif i < 30:
            ax.imshow(img_recA[i - 20].transpose(1, 2, 0))
        elif i < 40:
            ax.imshow(img_realB[i - 30].transpose(1, 2, 0))
        elif i < 50:
            ax.imshow(img_fakeA[i - 40].transpose(1, 2, 0))
        else:
            ax.imshow(img_recB[i - 50].transpose(1, 2, 0))

    plt.savefig('{}/samples_{:03d}'.format(savedir, epoch))
    # plt.show()
    plt.close()


def random_augmentation(image, crop_size, resize_size):
    # randomly choose crop size
    image = image.transpose(1, 2, 0)
    h, w, _ = image.shape

    # random cropping
    if crop_size != h:
        top = random.randint(0, h - crop_size - 1)
        left = random.randint(0, w - crop_size - 1)
        bottom = top + crop_size
        right = left + crop_size
        image = image[top:bottom, left:right, :]

    # random flipping
    if random.randint(0, 1):
        image = image[:, ::-1, :]

    # randomly choose resize size
    if resize_size != crop_size:
        cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_AREA)

    return image.transpose(2, 0, 1)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('out')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='# of epoch')
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--memory_size', '-m', type=int, default=500)
    parser.add_argument('--real_label', type=float, default=0.9)
    parser.add_argument('--fake_label', type=float, default=0.0)
    parser.add_argument('--block_num', type=int, default=6)
    parser.add_argument('--g_nobn', dest='g_bn', action='store_false', default=True)
    parser.add_argument('--d_nobn', dest='d_bn', action='store_false', default=True)
    parser.add_argument('--variable_size', action='store_true', default=False)
    parser.add_argument('--lambda_dis_real', type=float, default=0)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--lambda_', type=float, default=10)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # log directory
    out = datetime.datetime.now().strftime('%m%d%H')
    out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'visualize'), exist_ok=True)

    # hyper parameter
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    trainA = ImageDataset('horse2zebra/trainA', augmentation=True, image_size=256, final_size=args.size)
    trainB = ImageDataset('horse2zebra/trainB', augmentation=True, image_size=256, final_size=args.size)
    testA = ImageDataset('horse2zebra/testA', image_size=256, final_size=args.size)
    testB = ImageDataset('horse2zebra/testB', image_size=256, final_size=args.size)

    train_iterA = chainer.iterators.MultiprocessIterator(trainA, args.batch_size, n_processes=min(8, args.batch_size))
    train_iterB = chainer.iterators.MultiprocessIterator(trainB, args.batch_size, n_processes=min(8, args.batch_size))
    N = len(trainA)

    # genA convert B -> A, genB convert A -> B
    genA = Generator(block_num=args.block_num, bn=args.g_bn)
    genB = Generator(block_num=args.block_num, bn=args.g_bn)
    # disA discriminate realA and fakeA, disB discriminate realB and fakeB
    disA = Discriminator(bn=args.d_bn)
    disB = Discriminator(bn=args.d_bn)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        genA.to_gpu()
        genB.to_gpu()
        disA.to_gpu()
        disB.to_gpu()

    optimizer_genA = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_genB = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_disA = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_disB = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)

    optimizer_genA.setup(genA)
    optimizer_genB.setup(genB)
    optimizer_disA.setup(disA)
    optimizer_disB.setup(disB)

    # start training
    start = time.time()
    fake_poolA = np.zeros((args.memory_size, 3, args.size, args.size)).astype('float32')
    fake_poolB = np.zeros((args.memory_size, 3, args.size, args.size)).astype('float32')
    lambda_ = args.lambda_
    const_realA = np.asarray([testA.get_example(i) for i in range(10)])
    const_realB = np.asarray([testB.get_example(i) for i in range(10)])

    iterations = 0
    for epoch in range(args.epoch):

        if epoch > 100:
            decay_rate = 0.0002 / 100
            optimizer_genA.alpha -= decay_rate
            optimizer_genB.alpha -= decay_rate
            optimizer_disA.alpha -= decay_rate
            optimizer_disB.alpha -= decay_rate

        # train
        iter_num = N // args.batch_size
        for i in range(iter_num):

            # load real batch
            imagesA = train_iterA.next()
            imagesB = train_iterB.next()
            if args.variable_size:
                crop_size = np.random.choice([160, 192, 224, 256])
                resize_size = np.random.choice([160, 192, 224, 256])
                imagesA = [random_augmentation(image, crop_size, resize_size) for image in imagesA]
                imagesB = [random_augmentation(image, crop_size, resize_size) for image in imagesB]
            realA = chainer.Variable(genA.xp.asarray(imagesA, 'float32'))
            realB = chainer.Variable(genB.xp.asarray(imagesB, 'float32'))

            # load fake batch
            if iterations < args.memory_size:
                fakeA = genA(realB)
                fakeB = genB(realA)
                fakeA.unchain_backward()
                fakeB.unchain_backward()
            else:
                fake_imagesA = fake_poolA[np.random.randint(args.memory_size, size=args.batch_size)]
                fake_imagesB = fake_poolB[np.random.randint(args.memory_size, size=args.batch_size)]
                if args.variable_size:
                    fake_imagesA = [random_augmentation(image, crop_size, resize_size) for image in fake_imagesA]
                    fake_imagesB = [random_augmentation(image, crop_size, resize_size) for image in fake_imagesB]
                fakeA = chainer.Variable(genA.xp.asarray(fake_imagesA))
                fakeB = chainer.Variable(genA.xp.asarray(fake_imagesB))

            ############################
            # (1) Update D network
            ###########################
            # dis A
            y_realA = disA(realA)
            y_fakeA = disA(fakeA)
            loss_disA = (F.sum((y_realA - args.real_label) ** 2) + F.sum((y_fakeA - args.fake_label) ** 2)) \
                        / np.prod(y_fakeA.shape)

            # dis B
            y_realB = disB(realB)
            y_fakeB = disB(fakeB)
            loss_disB = (F.sum((y_realB - args.real_label) ** 2) + F.sum((y_fakeB - args.fake_label) ** 2)) \
                        / np.prod(y_fakeB.shape)

            # discriminate real A and real B not only realA and fakeA
            if args.lambda_dis_real > 0:
                y_realB = disA(realB)
                loss_disA += F.sum((y_realB - args.fake_label) ** 2) / np.prod(y_realB.shape)
                y_realA = disB(realA)
                loss_disB += F.sum((y_realA - args.fake_label) ** 2) / np.prod(y_realA.shape)

            # update dis
            disA.cleargrads()
            disB.cleargrads()
            loss_disA.backward()
            loss_disB.backward()
            optimizer_disA.update()
            optimizer_disB.update()

            ###########################
            # (2) Update G network
            ###########################

            # gan A
            fakeA = genA(realB)
            y_fakeA = disA(fakeA)
            loss_ganA = F.sum((y_fakeA - args.real_label) ** 2) / np.prod(y_fakeA.shape)

            # gan B
            fakeB = genB(realA)
            y_fakeB = disB(fakeB)
            loss_ganB = F.sum((y_fakeB - args.real_label) ** 2) / np.prod(y_fakeB.shape)

            # rec A
            recA = genA(fakeB)
            loss_recA = F.mean_absolute_error(recA, realA)

            # rec B
            recB = genB(fakeA)
            loss_recB = F.mean_absolute_error(recB, realB)

            # gen loss
            loss_gen = loss_ganA + loss_ganB + lambda_ * (loss_recA + loss_recB)
            # loss_genB = loss_ganB + lambda_ * (loss_recB + loss_recA)

            # update gen
            genA.cleargrads()
            genB.cleargrads()
            loss_gen.backward()
            # loss_genB.backward()
            optimizer_genA.update()
            optimizer_genB.update()

            # logging
            logger.plot('loss dis A', float(loss_disA.data))
            logger.plot('loss dis B', float(loss_disB.data))
            logger.plot('loss rec A', float(loss_recA.data))
            logger.plot('loss rec B', float(loss_recB.data))
            logger.plot('loss gen A', float(loss_gen.data))
            # logger.plot('loss gen B', float(loss_genB.data))
            logger.tick()

            # save to replay buffer
            fakeA = cuda.to_cpu(fakeA.data)
            fakeB = cuda.to_cpu(fakeB.data)
            for k in range(args.batch_size):
                fake_sampleA = fakeA[k]
                fake_sampleB = fakeB[k]
                if args.variable_size:
                    fake_sampleA = cv2.resize(fake_sampleA.transpose(1, 2, 0), (256, 256),
                                              interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                    fake_sampleB = cv2.resize(fake_sampleB.transpose(1, 2, 0), (256, 256),
                                              interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                fake_poolA[(iterations * args.batch_size) % args.memory_size + k] = fake_sampleA
                fake_poolB[(iterations * args.batch_size) % args.memory_size + k] = fake_sampleB

            iterations += 1
            progress_report(iterations, start, args.batch_size)

        if epoch % 5 == 0:
            logger.flush(out_dir)
            visualize(genA, genB, const_realA, const_realB, epoch=epoch, savedir=os.path.join(out_dir, 'visualize'))

            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.disA.model".format(epoch)), disA)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.disB.model".format(epoch)), disB)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.genA.model".format(epoch)), genA)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.genB.model".format(epoch)), genB)


if __name__ == '__main__':
    main()
