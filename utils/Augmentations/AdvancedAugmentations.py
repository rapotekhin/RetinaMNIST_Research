import numpy as np
import torch

class AdvancedAugmentations:

    def __init__(self, args: dict):
        """
        Summary: Initialization.
        Parameters:
            args: dict - config of training. See ./main.py.
        """
        self._args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mixup(self, x_batch: torch.Tensor,
                    y_batch: torch.Tensor,
                    alpha: float=1.0) -> tuple((torch.Tensor, torch.Tensor)):
        """
        Summary: This is Mixup implementation for One Hot Encoding targets.
        Parameters:
            x_batch: torch.Tensor - input tensor to the model with shape (batch_size, 3, height, width)
            y_batch: torch.Tensor - GT labels as a tensor with shape (batch_size, nb_classes)
            alpha: float - mixup coefficient
        Return:
            mixed_x: torch.Tensor - mixed input tensor to the model with shape (batch_size, 3, height, width)
            new_targets: torch.Tensor - mixed GT labels as a tensor with shape (batch_size, nb_classes)
        Reference:
            https://www.kaggle.com/vandalko/cnn-pytorch-kfold-mixup-lb0-644
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x_batch.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x_batch + (1 - lam) * x_batch[index, :]
        y_a, y_b = y_batch, y_batch[index]

        new_targets = lam * y_a + (1 - lam) * y_b

        return mixed_x, new_targets

    def cutmix(self, x_batch: torch.Tensor,
                     y_batch: torch.Tensor,
                     alpha: float) -> tuple((torch.Tensor, torch.Tensor)):
        """
        Summary: This is Cutmix implementation for One Hot Encoding targets.
        Parameters:
            x_batch: torch.Tensor - input tensor to the model with shape (batch_size, 3, height, width)
            y_batch: torch.Tensor - GT labels as a tensor with shape (batch_size, nb_classes)
            alpha: float - cutmix coefficient
        Return:
            new_data: torch.Tensor - cutmixed input tensor to the model with shape (batch_size, 3, height, width)
            new_targets: torch.Tensor - cutmixed GT labels as a tensor with shape (batch_size, nb_classes)
        Reference:
            https://www.kaggle.com/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix
        """
        indices = torch.randperm(x_batch.size(0))
        shuffled_target = y_batch[indices]

        lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x_batch.size(), lam)
        new_data = x_batch.clone()
        new_data[:, :, bby1: bby2, bbx1: bbx2] = x_batch[indices, :, bby1: bby2, bbx1: bbx2]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_batch.size()[-1] * x_batch.size()[-2]))

        new_targets = lam * y_batch + (1 - lam) * shuffled_target

        return new_data, new_targets

    def _rand_bbox(self, size: tuple, lam: np.ndarray) -> tuple((np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray)):
        """
        Summary: This function for selecting Cutout bounding box.
        Parameters:
            size: tuple - size of the tensor
            lam: np.ndarray - array with cutmix coefficients for each example 
                              in the batch with shape (batch_size, 1)
        Return:
            bbx1: np.ndarray - array with left-top X coordinates for cutout bounding box 
                               for every example from batch (batch_size, 1)
            bby1: np.ndarray - array with left-top Y coordinates for cutout bounding box 
                               for every example from batch (batch_size, 1)
            bbx2: np.ndarray - array with right-botton X coordinates for cutout bounding box 
                               for every example from batch (batch_size, 1)
            bby2: np.ndarray - array with right-botton Y coordinates for cutout bounding box 
                               for every example from batch (batch_size, 1)
        Reference:
            https://www.kaggle.com/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix
        """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2