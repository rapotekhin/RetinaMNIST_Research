from collections import Counter

import torch
import numpy as np
import cv2

import medmnist
from medmnist.info import INFO, DEFAULT_ROOT

import albumentations as A
from albumentations.pytorch import ToTensorV2

info = INFO['retinamnist']
DataClass = getattr(medmnist, info['python_class'])

class RetinaMNISTDataset(DataClass):
    """
    Summary: This class realize some methods for download and using RetinaMNIST dataset.
    Features: 
        Augmentations - using 'albumentation' library for apply basic augmentation,
                        and using custom 'cutmix' and 'mixup' advanced augmentations
        Label Smoothing - using classic label smothing and custom method (see utils.Datasets.RetinaMNISTDataset)
        Inverse Labels Destribution - calculate inverse labels destribution for Focal Loss
    References:
        https://medmnist.com/
    """
    def __init__(self, split: str,
                       augment: bool=False,
                       download: bool=False,
                       as_rgb: bool=True,
                       root: str=DEFAULT_ROOT,
                       args: dict={}) -> None:
        """
        Summary: Initialization.
        Parameters:
            split: str - 'train', 'val' or 'test', select subset
            augment: bool - set True if you want to use data augmentation
            download: bool - set True if need download the dataset
            as_rgb: bool - set True if need convert images to RGB format
            root: str - default root path
            args: dict - config of training. See ./main.py.
        """
        super(RetinaMNISTDataset, self).__init__(split, None, None, download, as_rgb, root)

        self.args = args
        self.augment = augment
        self.nb_classes = self.args['nb_classes']
        self.transform = self._get_transforms()


    def __getitem__(self, index: int) -> tuple((torch.Tensor, torch.Tensor)):
        """
        Summary: Get one example of the dataset
        Parameters:
            index: int - index of the image and label in dataset
        Return:
            img: torch.Tensor - image as a PyTorch Tensor
            target: torch.Tensor - class-label as a PyTorch Tensor
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        if self.as_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        img = augmented['image']

        target = self._numpy_one_hot(target, num_classes=self.nb_classes)

        if self.args['label_smoothing'] == 'norm' and self.split == 'train':
            target = self._label_smoothing_norm(target)
        elif self.args['label_smoothing'] == 'classic' and self.split == 'train':
            target = self._label_smoothing_classic(target)
        else:
            pass
        
        target = torch.Tensor(target).squeeze().to(torch.float)
        return img, target

    def get_labels_inverse_destribution(self) -> np.ndarray:
        """
        Summary: Calculate inverse labels destribution for Focal Loss
        """
        cnt = Counter(self.labels.flatten().tolist())
        destribution = np.array(list(cnt.values()), dtype=int)
        labels_hist, _ = np.histogram(destribution, bins=self.nb_classes)

        return 1 - labels_hist / np.sum(labels_hist)

    def _numpy_one_hot(self, value: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Summary: Compute One Hot Vector from single value.
        Parameters:
            value: np.ndarray - labels as single number, vector with shape (batch_size,)
            num_classes: int - number of class-labels in the dataset
        Return:
            np.ndarray - One Hot Vector with shape (batch_size, num_classes)
        """
        return np.squeeze(np.eye(num_classes)[value.reshape(-1)])

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
        Summary: Custom label smoothing which I use 1D Convolution for blurring the One Hot Vector.
                 This function is a prototype and it needs to do later. 
        Parameters:
            targets: np.ndarray - Ground Truth one hot encoding targets with shape (batch_size, nb_classes)
        Return:
            targets: np.ndarray - smooth Ground Truth vector with shape (batch_size, nb_classes)
        """
        return np.convolve(targets, [0.05, 0.9, 0.05], 'same')

    def _get_transforms(self) -> A.Compose:
        """
        Summary: Apply base augmentations and transformation to the image.
        """
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