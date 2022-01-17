
import torch
import torch.nn as nn
import numpy as np

class WeightedFocalLoss(nn.Module):
    """
    Summary: Weighted version of Focal Loss
    Parameters:
        alpha: np.ndarray or int - inverse normed destribution of classes in the dataset with shape (nb_classes, ) 
                                   If args.use_FL_alpha = True in config than alpha=1 and doesn't change loss
        gamma: float - gamma parameter in classic Focal Loss
    """
    def __init__(self, alpha: np.ndarray or int, gamma: float=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cpu()
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Summary: Forward propagation of the Focal Loss
        Parameters:
            inputs: torch.Tensor - Model output as a One Hot Encoding, shape (batch_size, nb_classes)
            targets: torch.Tensor - GT labels as a One Hot Encoding, shape (batch_size, nb_classes)
        Return:
            F_loss: torch.Tensor - mean loss
        '''
        BCE_loss = nn.BCELoss()(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = 100*at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()