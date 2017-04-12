import argparse
import os, sys
import numpy as np
import datetime
import time
import pickle

import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

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
    x_fakeA = genA(chainer.Variable(genA.xp.asarray(realA, 'float32')))
    img_genA = ((cuda.to_cpu(x_fakeA.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    img_realB = ((realB + 1) * 127.5).clip(0, 255).astype(np.uint8)
    x_fakeB = genB(chainer.Variable(genB.xp.asarray(realB, 'float32')))
    img_genB = ((cuda.to_cpu(x_fakeB.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(40):
        ax = fig.add_subplot(4, 10, i + 1, xticks=[], yticks=[])
        if i < 10:
            ax.imshow(img_realA[i].transpose(1, 2, 0))
        elif i < 20:
            ax.imshow(img_genA[i - 10].transpose(1, 2, 0))
        elif i < 30:
            ax.imshow(img_realB[i - 20].transpose(1, 2, 0))
        else:
            ax.imshow(img_genB[i - 30].transpose(1, 2, 0))

    plt.savefig('{}/samples_{:03d}'.format(savedir, epoch))
    # plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('out')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='# of epoch')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--memory_size', '-m', type=int, default=50)
    parser.add_argument('--real_label', type=float, default=0.9)
    parser.add_argument('--fake_label', type=float, default=0)
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

    datapath = ['horse2zebra/trainA', 'horse2zebra/trainB', 'horse2zebra/testA', 'horse2zebra/testB']
    trainA, trainB, testA, testB = [ImageDataset(d) for d in datapath]

    train_iterA = chainer.iterators.MultiprocessIterator(trainA, args.batch_size)
    train_iterB = chainer.iterators.MultiprocessIterator(trainB, args.batch_size)
    N = len(trainA)

    # genA convert B -> A, genB convert A -> B
    genA = Generator()
    genB = Generator()
    # disA discriminate realA and fakeA, disB discriminate realB and fakeB
    disA = Discriminator()
    disB = Discriminator()

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        genA.to_gpu()
        genB.to_gpu()
        disA.to_gpu()
        disB.to_gpu()

    optimizer_genA = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_genB = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_disA = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_disB = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)

    optimizer_genA.setup(genA)
    optimizer_genB.setup(genB)
    optimizer_disA.setup(disA)
    optimizer_disB.setup(disB)

    # start training
    start = time.time()
    fake_poolA = np.zeros((args.memory_size, 3, 256, 256)).astype('float32')
    fake_poolB = np.zeros((args.memory_size, 3, 256, 256)).astype('float32')
    lambda_ = args.lambda_
    const_realA = np.asarray([testA.get_example(i) for i in range(10)])
    const_realB = np.asarray([testB.get_example(i) for i in range(10)])

    iterations = 0
    for epoch in range(args.epoch):

        if epoch in [5, 10, 15]:
            optimizer_genA.alpha *= 0.5
            optimizer_genB.alpha *= 0.5
            optimizer_disA.alpha *= 0.5
            optimizer_disB.alpha *= 0.5

        # train
        iter_num = N // args.batch_size
        for i in range(iter_num):
            realA = chainer.Variable(genA.xp.asarray(train_iterA.next(), 'float32'))
            realB = chainer.Variable(genA.xp.asarray(train_iterB.next(), 'float32'))
            if iterations < args.memory_size:
                fakeA = genA(realB, train=False)
                fakeB = genB(realA, train=False)
                fakeA.unchain_backward()
                fakeB.unchain_backward()
            else:
                fakeA = chainer.Variable(genA.xp.asarray(fake_poolA[np.random.randint(args.memory_size, size=args.batch_size)]))
                fakeB = chainer.Variable(genA.xp.asarray(fake_poolB[np.random.randint(args.memory_size, size=args.batch_size)]))

            ############################
            # (1) Update D network
            ###########################
            # dis A
            y_realA = disA(realA)
            y_fakeA = disA(fakeA)
            loss_disA = 0.5 * (F.sum((y_realA - args.real_label) ** 2) + F.sum((y_fakeA - args.fake_label) ** 2)) \
                        / args.batch_size

            # dis B
            y_realB = disB(realB)
            y_fakeB = disB(fakeB)
            loss_disB = 0.5 * (F.sum((y_realB - args.real_label) ** 2) + F.sum((y_fakeB - args.fake_label) ** 2)) \
                        / args.batch_size

            # update dis
            disA.cleargrads()
            loss_disA.backward()
            optimizer_disA.update()

            disB.cleargrads()
            loss_disB.backward()
            optimizer_disB.update()

            ###########################
            # (2) Update G network
            ###########################

            # gan A
            fakeA = genA(realB, train=True)
            y_fakeA = disA(fakeA)
            loss_ganA = 0.5 * F.sum((y_fakeA - args.real_label) ** 2) / args.batch_size

            # gan B
            fakeB = genB(realA, train=True)
            y_fakeB = disB(fakeB)
            loss_ganB = 0.5 * F.sum((y_fakeB - args.real_label) ** 2) / args.batch_size

            # rec A
            recA = genA(fakeB)
            loss_recA = F.mean_absolute_error(recA, realA)

            # rec B
            recB = genB(fakeA)
            loss_recB = F.mean_absolute_error(recB, realB)

            loss_genA = loss_ganA + lambda_ * (loss_recA + loss_recB)
            loss_genB = loss_ganB + lambda_ * (loss_recB + loss_recA)

            genA.cleargrads()
            loss_genA.backward()
            optimizer_genA.update()

            genB.cleargrads()
            loss_genB.backward()
            optimizer_genB.update()

            # logging
            logger.plot('loss dis A', float(loss_disA.data))
            logger.plot('loss dis B', float(loss_disB.data))
            logger.plot('loss rec A', float(loss_recA.data))
            logger.plot('loss rec B', float(loss_recB.data))
            logger.plot('loss gen A', float(loss_genA.data))
            logger.plot('loss gen B', float(loss_genB.data))
            logger.tick()

            # save to replay buffer
            fakeA = cuda.to_cpu(fakeA.data)
            fakeB = cuda.to_cpu(fakeB.data)
            for k in range(args.batch_size):
                fake_poolA[(iterations * args.batch_size) % args.memory_size + k] = fakeA[k]
                fake_poolB[(iterations * args.batch_size) % args.memory_size + k] = fakeB[k]

            iterations += 1
            progress_report(iterations, start, args.batch_size)

        logger.flush(out_dir)
        visualize(genA, genB, const_realA, const_realB, epoch=epoch, savedir=os.path.join(out_dir, 'visualize'))

        if epoch % 5 == 0:
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.disA.model".format(epoch)), disA)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.disB.model".format(epoch)), disB)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.genA.model".format(epoch)), genA)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.genB.model".format(epoch)), genB)


if __name__ == '__main__':
    main()
