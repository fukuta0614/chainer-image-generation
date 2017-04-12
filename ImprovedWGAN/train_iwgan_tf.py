import argparse
import os, sys
import numpy as np
import datetime
import time
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iwgan
from dataset import CelebA

try:
    import tensorflow as tf

    use_tensorboard = True
except:
    print('tensorflow is not installed')
    use_tensorboard = False


def progress_report(count, start_time, batchsize, emd, norm):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r{} updates ({} samples) time: {} ({:.2f} samples/sec) emd : {:.4f} norm : {:.4f}'.format(
            count, count * batchsize, str(datetime.timedelta(seconds=duration)).split('.')[0], throughput, emd, norm
        )
    )


def visualize(gen, epoch, savedir, batch_size=36, image_type='sigmoid'):
    z = chainer.Variable(gen.xp.asarray(gen.make_hidden(batch_size)), volatile=True)
    x_fake = gen(z, train=False)
    if image_type == 'sigmoid':
        img_gen = ((cuda.to_cpu(x_fake.data)) * 255).clip(0, 255).astype(np.uint8)
    else:
        img_gen = ((cuda.to_cpu(x_fake.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(36):
        ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(img_gen[i].transpose(1, 2, 0))
    fig.savefig('{}/generate_{:03d}'.format(savedir, epoch))
    # plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='# of epoch')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--g_hidden', type=int, default=128)
    parser.add_argument('--g_arch', type=int, default=1)
    parser.add_argument('--g_activate', type=str, default='sigmoid')
    parser.add_argument('--g_channel', type=int, default=512)
    parser.add_argument('--d_arch', type=int, default=1)
    parser.add_argument('--d_iters', type=int, default=5)
    parser.add_argument('--d_channel', type=int, default=512)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--initial_iter', type=int, default=10)
    parser.add_argument('--resume', default='')
    parser.add_argument('--out', default='')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # log directory
    out = datetime.datetime.now().strftime('%m%d%H')
    if args.out:
        out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'visualize'), exist_ok=True)

    # hyper parameter
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    # tensorboard
    if use_tensorboard:
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        summary_dir = os.path.join(out_dir, "summaries")
        loss_ = tf.placeholder(tf.float32)
        gen_loss_summary = tf.scalar_summary('gen_loss', loss_)
        dis_loss_summary = tf.scalar_summary('dis_loss', loss_)
        summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

    # load celebA
    dataset = CelebA(image_type=args.g_activate)
    train_iter = chainer.iterators.MultiprocessIterator(dataset, args.batch_size)

    gen = iwgan.Generator(n_hidden=args.g_hidden, activate=args.g_activate, ch=args.g_channel)
    dis = iwgan.Discriminator(ch=args.d_channel)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    optimizer_gen = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_dis = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)

    optimizer_gen.setup(gen)
    optimizer_dis.setup(dis)

    # start training
    start = time.time()
    gen_iterations = 0
    lambda_gp = args.lambda_gp

    for epoch in range(args.epoch):

        # train
        sum_L_gen = []
        sum_L_dis = []

        i = 0
        while i < len(dataset) // args.batch_size:

            # tips for critic reach optimality
            if gen_iterations < args.initial_iter or gen_iterations % 500 == 0:
                d_iters = 10
            else:
                d_iters = args.d_iters

            ############################
            # (1) Update D network
            ###########################
            j = 0
            while j < d_iters:
                batch = train_iter.next()
                x = chainer.Variable(gen.xp.asarray([b[0] for b in batch], 'float32'))
                # attr = chainer.Variable(gen.xp.asarray([b[1] for b in batch], 'int32'))

                # real
                y_real = dis(x)

                # fake
                z = chainer.Variable(gen.xp.asarray(gen.make_hidden(args.batch_size)), volatile=True)
                x_fake = gen(z)
                x_fake.volatile = False
                y_fake = dis(x_fake)

                # calc gradient penalty
                epsilon = gen.xp.random.uniform(0, 1, (args.batch_size, 1, 1, 1)).astype('float32')
                x_hat = chainer.Variable(epsilon * x.data + (1 - epsilon) * x_fake.data)
                dis(x_hat).backward()
                L_gp = (1 - F.sqrt(F.sum(x_hat.grad ** 2, axis=(1, 2, 3)))) ** 2  / args.batch_size

                # calc EMD
                L_dis_origin = - (y_real - y_fake)
                L_dis = L_dis_origin + lambda_gp * L_gp
                dis.cleargrads()
                L_dis.backward()
                optimizer_dis.update()

                j += 1
                i += 1

            ###########################
            # (2) Update G network
            ###########################
            z = chainer.Variable(gen.xp.asarray(gen.make_hidden(args.batch_size)))
            x_fake = gen(z)
            y_fake = dis(x_fake)
            L_gen = - y_fake
            gen.cleargrads()
            L_gen.backward()
            optimizer_gen.update()

            gen_iterations += 1

            emd = float(-L_dis_origin.data)
            norm = float(L_gp.data)
            l_gen = float(L_gen.data)
            sum_L_dis.append(emd)
            sum_L_gen.append(l_gen)

            progress_report(epoch * len(dataset) // args.batch_size + i, start, args.batch_size, emd, norm)

            if use_tensorboard:
                summary = sess.run(gen_loss_summary, feed_dict={loss_: l_gen})
                summary_writer.add_summary(summary, gen_iterations)
                summary = sess.run(dis_loss_summary, feed_dict={loss_: emd})
                summary_writer.add_summary(summary, gen_iterations)

        log = 'gen loss={:.5f}, dis loss={:.5f}'.format(np.mean(sum_L_gen), np.mean(sum_L_dis))
        print('\n' + log)
        with open(os.path.join(out_dir, "log"), 'a+') as f:
            f.write(log + '\n')

        if epoch % 5 == 0:
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.dis.model".format(epoch)), dis)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.gen.model".format(epoch)), gen)

        visualize(gen, epoch=epoch, savedir=os.path.join(out_dir, 'visualize'), image_type=args.g_activate)


if __name__ == '__main__':
    main()
