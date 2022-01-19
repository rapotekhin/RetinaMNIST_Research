import torch
import numpy as np

from PIL import Image
import cv2

from torchvision import transforms
from medmnist.info import DEFAULT_ROOT

from utils.Datasets.RetinaMNISTDataset import RetinaMNISTDataset

class RetinaMnistDatasetOrdinalRegression(RetinaMNISTDataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self, split: str,
                       augment: bool=False,
                       download: bool=False,
                       as_rgb: bool=True,
                       root: str=DEFAULT_ROOT,
                       args: dict={}) -> None:

        super(RetinaMnistDatasetOrdinalRegression, self).__init__(split, augment, download, as_rgb, root, args)

        self.transform = self._get_transforms()

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(np.uint8(img)).convert('RGB')

        img = self.transform(img)
        levels = [[1]*label_i + [0]*(self.nb_classes - 1 - label_i) for label_i in label]
        levels = torch.tensor(levels, dtype=torch.float32)

        return img, label, levels

    def __len__(self):
        return self.labels.shape[0]

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([transforms.Resize((28, 28)),
                                #    transforms.RandomCrop((120, 120)),
                                   transforms.ToTensor()])
