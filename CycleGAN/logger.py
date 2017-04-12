import numpy as np
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]


def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush(out_dir):
    print("\niter {}".format(_iter[0]))
    for name, vals in sorted(_since_last_flush.items()):
        print(" {}\t{:.5f}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(out_dir, name.replace(' ', '_') + '.jpg'))

    _since_last_flush.clear()

    with open(os.path.join(out_dir, 'log.pkl'), 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
