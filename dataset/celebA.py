import chainer
import os
import numpy as np
from PIL import Image


class CelebA(chainer.dataset.DatasetMixin):
    def __init__(self, dataset_home='/home/mil/fukuta/datasets/', image_size=64, image_type='sigmoid', nodivide=False, type='train'):
        self.image_type = image_type
        self.nodivide = nodivide
        self.name = 'celeba'
        self.n_imgs = 202599
        self.n_attrs = 40
        self.image_size = image_size
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, self.name + '.npz')
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba')
        (self.train_idxs, self.val_idxs, self.test_idxs, self.attribute_names,
         self.attributes) = self._load()

    def __len__(self):
        return self.n_imgs

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return (dic['train_idxs'], dic['val_idxs'], dic['test_idxs'],
                    dic['attribute_names'][()], dic['attributes'])

    def get_image(self, idx):
        img_path = os.path.join(self.img_dir, '%.6d.jpg' % (idx + 1))
        return Image.open(img_path)

    def get_attributes(self, idx):
        return self.attributes[idx]

    def get_example(self, i):
        image = self.get_image(i)
        attr = self.get_attributes(i)

        offset_x = np.random.randint(8) + 13
        offset_y = np.random.randint(8) + 33
        w = 144
        h = 144
        image = np.asarray(image.convert('RGB').
                           crop((offset_x, offset_y, offset_x + w, offset_y + h)).
                           resize((self.image_size, self.image_size)))

        image = image.astype(np.float32).transpose((2, 0, 1))

        # pre-process
        if not self.nodivide:
            if self.image_type == 'tanh':
                image = image / 127.5 - 1
            elif self.image_type == 'sigmoid':
                image /= 255.
            else:
                raise ValueError('invalid image type')

        return image, attr
