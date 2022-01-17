import os
import numpy as np
import cv2
from collections import Counter

import medmnist
from medmnist import INFO
from medmnist.info import INFO, DEFAULT_ROOT

import albumentations as A
from albumentations.pytorch import ToTensorV2

info = INFO['retinamnist']
DataClass = getattr(medmnist, info['python_class'])

class RetinaMNISTDataset(DataClass):

    def __init__(self, split,
                       augment=False,
                       target_transform=None,
                       download=False,
                       as_rgb=False,
                       root=DEFAULT_ROOT,
                       args=None) -> None:
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: bool
        :param target_transform: bool
        '''

        self.args = args
        self.info = INFO[self.flag]
        self.nb_classes = self.args['nb_classes']

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found. ' +
                               ' You can set `download=True` to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.augment = augment
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == 'train':
            self.imgs = npz_file['train_images']
            self.labels = npz_file['train_labels']
        elif self.split == 'val':
            self.imgs = npz_file['val_images']
            self.labels = npz_file['val_labels']
        elif self.split == 'test':
            self.imgs = npz_file['test_images']
            self.labels = npz_file['test_labels']
        else:
            raise ValueError

        self.transform = self._get_transforms()


    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        img = augmented['image']


        if self.target_transform is not None:
            target = self.target_transform(target)

        target = self._numpy_one_hot(target, num_classes=self.nb_classes)

        if self.args['label_smoothing'] == 'norm' and self.split == 'train':
            target = self._label_smoothing_norm(target)
        elif self.args['label_smoothing'] == 'classic' and self.split == 'train':
            target = self._label_smoothing_classic(target)
        else:
            pass

        return img, target

    def get_labels_inverse_destribution(self) -> np.ndarray:
        cnt = Counter(self.labels.flatten().tolist())
        destribution = np.array(list(cnt.values()), dtype=int)
        labels_hist, _ = np.histogram(destribution, bins=self.nb_classes)

        return 1 - labels_hist / np.sum(labels_hist)

    def _numpy_one_hot(self, a, num_classes) -> np.ndarray:
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def _label_smoothing_classic(self, targets: np.ndarray, smoothing: float = 0.2) -> np.ndarray:
        """
        Summary: Function for apply classic label smoothing algorithm.
        Parameters:
            targets: np.ndarray - Ground Truth one hot encoding targets with shape (batch_size, nb_classes)
            nb_classes: int - number of classes in dataset
            smoothing: float - if smoothing == 0, it's one-hot method; if 0 < smoothing < 1, it's smooth method
        Return:
            targets: np.ndarray - smooth Ground Truth vector with shape (batch_size, nb_classes)
        References:
            https://arxiv.org/abs/1906.02629
        """
        assert 0 <= smoothing < 1

        targets *= 1.0 - smoothing
        targets += smoothing / (self.nb_classes - 1)

        return targets

    def _label_smoothing_norm(self, targets: np.ndarray) -> np.ndarray:
        """
        Summary: Function for apply classic label smoothing algorithm.
        Parameters:
            targets: np.ndarray - Ground Truth one hot encoding targets with shape (batch_size, nb_classes)
        Return:
            targets: np.ndarray - smooth Ground Truth vector with shape (batch_size, nb_classes)
        """

        return np.convolve(targets, [0.05, 0.9, 0.05], 'same')

    def _get_transforms(self) -> A.Compose:

        if self.split == 'train' and self.augment:
            return A.Compose([

                A.Resize(self.args['img_size'], self.args['img_size'], p=1.0),

                A.Transpose(p=self.args['Transpose']),
                A.HorizontalFlip(p=self.args['HorizontalFlip']),
                A.VerticalFlip(p=self.args['VerticalFlip']),
                A.GridDistortion(p=self.args['GridDistortion']),
                A.ShiftScaleRotate(p=self.args['ShiftScaleRotate']), 
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, 
                                     p=self.args['HueSaturationValue']),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), 
                                           p=self.args['RandomBrightnessContrast']),

                A.Normalize(p=1.0),
                ToTensorV2(),
            ])

        else:
            return A.Compose(
                [
                    A.Resize(self.args['img_size'], self.args['img_size'], p=1),
                    A.Normalize(p=1.0),
                    ToTensorV2(),
                ])