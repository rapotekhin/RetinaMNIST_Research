import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalLoss(nn.Module):
    """
    Summary: Weighted version of Focal Loss
    Parameters:
        alpha: np.ndarray or int - inverse normed destribution of classes in the dataset with shape (nb_classes, ) 
                                   If args.use_FL_alpha = True in config than alpha=1 and doesn't change loss
        gamma: float - gamma parameter in classic Focal Loss
    """
    def __init__(self):
        super(OrdinalLoss, self).__init__()


    def forward(self, logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        '''
        Summary: Forward propagation of the Ordinal Loss
        Parameters:
            logits: torch.Tensor - Model output as a One Hot Encoding, shape (batch_size, nb_classes)
            levels: torch.Tensor - GT labels as a levels, shape (batch_size, nb_classes)
        Return:
            torch.Tensor - mean loss
        '''
        val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels)), dim=1))
        return torch.mean(val)