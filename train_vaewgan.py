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

from models import wgan
from dataset import CelebA

try:
    import tensorflow as tf
    use_tensorboard = True
except:
    print('tensorflow is not installed')
    use_tensorboard = False


def progress_report(count, start_time, batchsize, emd):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r{} updates ({} samples) time: {} ({:.2f} samples/sec) emd : {:.5f}'.format(
            count, count * batchsize, str(datetime.timedelta(seconds=duration)).split('.')[0], throughput, emd
        )
    )


def visualize(gen, epoch, savedir, batch_size=64):

    z = chainer.Variable(gen.xp.asarray(gen.make_hidden(batch_size)), volatile=True)
    x_fake = gen(z, train=False)
    img_gen = ((cuda.to_cpu(x_fake.data)) * 255).clip(0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(img_gen[i].transpose(1, 2, 0))
    fig.savefig('{}/generate_{:03d}'.format(savedir, epoch))
    # plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='# of epoch')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--g_hidden', type=int, default=128)
    parser.add_argument('--d_iters', type=int, default=5)
    parser.add_argument('--initial_iter', type=int, default=10)
    parser.add_argument('--d_clip', type=float, default=0.01)
    parser.add_argument('--out', default='')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # log directory
    out = datetime.datetime.now().strftime('%m%d')
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
        enc_loss_summary = tf.scalar_summary('enc_loss', loss_)
        rec_loss_summary = tf.scalar_summary('rec_loss', loss_)
        summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

    # load celebA
    dataset = CelebA()
    train_iter = chainer.iterators.MultiprocessIterator(dataset, args.batch_size)

    gen = wgan.Generator(n_hidden=args.g_hidden)
    dis = wgan.Discriminator()
    enc = wgan.Encoder()


    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        enc.to_gpu()

    optimizer_gen = chainer.optimizers.RMSprop(lr=0.00005)
    optimizer_dis = chainer.optimizers.RMSprop(lr=0.00005)
    optimizer_enc = chainer.optimizers.RMSprop(lr=0.00005)

    optimizer_gen.setup(gen)
    optimizer_dis.setup(dis)
    optimizer_enc.setup(enc)

    # optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    # optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))
    # optimizer_enc.add_hook(chainer.optimizer.WeightDecay(0.00001))

    # start training
    start = time.time()
    train_count = 0
    gen_iterations = 0
    for epoch in range(args.epoch):

        # train
        sum_L_gen = []
        sum_L_dis = []
        sum_L_enc = []
        sum_L_rec = []

        i = 0
        while i < len(dataset) // args.batch_size:

            # tips for critic reach optimality
            if gen_iterations < args.initial_iter or gen_iterations % 500 == 0:
                d_iters = 100
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
                z = chainer.Variable(gen.xp.asarray(gen.make_hidden(args.batch_size)))
                # z = chainer.Variable(gen.xp.random.normal(0, 1, (args.batchsize, args.n_hidden)).astype(np.float32))

                # real image
                y_real, _, _ = dis(x)

                # fake image from random noize
                x_fake = gen(z)
                y_fake, _, _ = dis(x_fake)

                # fake image from reconstruction
                mu_z, ln_var_z = enc(x)
                z_rec = F.gaussian(mu_z, ln_var_z)
                x_rec = gen(z_rec)
                y_rec, _, _ = dis(x_rec)

                L_dis = - (y_real - 0.5 * y_fake - 0.5 * y_rec)
                # print(j, -L_dis.data)

                dis.cleargrads()
                L_dis.backward()
                optimizer_dis.update()

                dis.clip_weight(clip=args.d_clip)

                j += 1
                i += 1

            ###########################
            # (2) Update Enc and Dec network
            ###########################
            batch = train_iter.next()
            x = chainer.Variable(gen.xp.asarray([b[0] for b in batch], 'float32'))
            z = chainer.Variable(gen.xp.asarray(gen.make_hidden(args.batch_size)))

            # real image
            y_real, l2_real, l3_real = dis(x)

            # fake image from random noize
            x_fake = gen(z)
            y_fake, _, _ = dis(x_fake)

            # fake image from reconstruction
            mu_z, ln_var_z = enc(x)
            z_rec = F.gaussian(mu_z, ln_var_z)
            x_rec = gen(z_rec)
            y_rec, l2_rec, l3_rec = dis(x_rec)

            L_rec = F.mean_squared_error(l2_real, l2_rec) * l2_real.data.shape[2] * l2_real.data.shape[3]
            L_prior = F.gaussian_kl_divergence(mu_z, ln_var_z) / args.batch_size
            L_gan = - 0.5 * y_fake - 0.5 * y_rec

            L_gen = L_gan + L_rec
            L_enc = L_rec + L_prior

            gen.cleargrads()
            L_gen.backward()
            optimizer_gen.update()

            enc.cleargrads()
            L_enc.backward()
            optimizer_enc.update()

            gen_iterations += 1

            emd = float(-L_dis.data)
            sum_L_dis.append(emd)
            sum_L_gen.append(float(L_gen.data))
            sum_L_enc.append(float(L_enc.data))
            sum_L_rec.append(float(L_rec.data))

            progress_report(epoch * len(dataset) + i, start, args.batch_size, emd)

        log = 'gen loss={:.5f}, dis loss={:.5f} enc loss={:.5f} rec loss={:.5f}'\
            .format(np.mean(sum_L_gen), np.mean(sum_L_dis), np.mean(sum_L_enc), np.mean(sum_L_rec))
        print('\n' + log)
        with open(os.path.join(out_dir, "log"), 'a+') as f:
            f.write(log + '\n')

        if use_tensorboard:
            summary = sess.run(gen_loss_summary, feed_dict={loss_: np.mean(sum_L_gen)})
            summary_writer.add_summary(summary, epoch)
            summary = sess.run(dis_loss_summary, feed_dict={loss_: np.mean(sum_L_dis)})
            summary_writer.add_summary(summary, epoch)
            summary = sess.run(enc_loss_summary, feed_dict={loss_: np.mean(sum_L_enc)})
            summary_writer.add_summary(summary, epoch)
            summary = sess.run(rec_loss_summary, feed_dict={loss_: np.mean(sum_L_rec)})
            summary_writer.add_summary(summary, epoch)

        if epoch % 5 == 0:
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.dis.model".format(epoch)), dis)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.gen.model".format(epoch)), gen)
            serializers.save_hdf5(os.path.join(out_dir, "models", "{:03d}.enc.model".format(epoch)), enc)

        visualize(gen, epoch=epoch, savedir=os.path.join(out_dir, 'visualize'))


if __name__ == '__main__':
    main()
