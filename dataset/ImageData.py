import chainer
import os
import numpy as np
from PIL import Image
import cv2
import random


class ImageDataset(chainer.dataset.DatasetMixin):
    dataset_home = '/home/mil/fukuta/datasets/'

    def __init__(self, name, preprocess=1, augmentation=False, variable_size=False, image_size=256, final_size=128):
        self.augmentation = augmentation
        self.variable_size = variable_size
        self.image_size = image_size
        self.final_size = final_size
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

        if not self.variable_size:
            if self.augmentation:
                # Randomly crop a region and flip the image
                crop_size = np.random.choice([192, 224, 256])
                h, w, _ = image.shape
                top = random.randint(0, h - crop_size)
                left = random.randint(0, w - crop_size)
                bottom = top + crop_size
                right = left + crop_size
                image = image[top:bottom, left:right, :]
                if random.randint(0, 1):
                    image = image[:, ::-1, :]

            image = cv2.resize(image, (self.final_size, self.final_size), interpolation=cv2.INTER_AREA)

        # pre-process
        image = image.astype(np.float32).transpose((2, 0, 1))
        if self.preprocess == 0:
            pass
        elif self.preprocess == 1:
            image = image / 127.5 - 1
        elif self.preprocess == 2:
            image /= 255.
        else:
            raise ValueError('invalid image type')

        return image
