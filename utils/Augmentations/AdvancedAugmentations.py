import numpy as np
import torch


class AdvancedAugmentations:

    def __init__(self, args: dict):
        self._args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mixup(self, x, y, alpha=1.0):
        # https://www.kaggle.com/vandalko/cnn-pytorch-kfold-mixup-lb0-644
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        new_targets = lam*y_a + (1-lam)*y_b

        return mixed_x, new_targets

    def cutmix(self, x_batch, y_batch, alpha):
        # https://www.kaggle.com/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix
        indices = torch.randperm(x_batch.size(0))
        shuffled_data = x_batch[indices]
        shuffled_target = y_batch[indices]

        lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x_batch.size(), lam)
        new_data = x_batch.clone()
        new_data[:, :, bby1:bby2, bbx1:bbx2] = x_batch[indices, :, bby1:bby2, bbx1:bbx2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_batch.size()[-1] * x_batch.size()[-2]))

        new_targets = lam*y_batch + (1-lam)*shuffled_target

        return new_data, new_targets

    def _rand_bbox(self, size, lam):
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