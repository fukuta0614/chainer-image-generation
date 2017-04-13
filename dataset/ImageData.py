import chainer
import os
import numpy as np
from PIL import Image


class ImageDataset(chainer.dataset.DatasetMixin):
    dataset_home = '/home/mil/fukuta/datasets/'

    def __init__(self, name, preprocess=1):
        self.preprocess = preprocess
        self.name = name
        self.data_dir = os.path.join(self.dataset_home, self.name)
        self.data = [os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir)]
        self.n_imgs = len(self.data)

    def __len__(self):
        return self.n_imgs

    def get_example(self, i):
        img_path = self.data[i]
        image = np.asarray(Image.open(img_path))
        image = image.astype(np.float32).transpose((2, 0, 1))

        # pre-process
        if self.preprocess == 0:
            pass
        elif self.preprocess == 1:
            image = image / 127.5 - 1
        elif self.preprocess == 2:
            image /= 255.
        else:
            raise ValueError('invalid image type')

        return image
