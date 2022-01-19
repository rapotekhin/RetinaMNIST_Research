from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from medmnist.info import INFO, DEFAULT_ROOT

def getACC_ordinal(y_true, y_pred):
    '''Accuracy metrics
    :param y_true: the ground truth labels, shape: (n_samples,)
    :param y_pred: the predicted score of each class, shape: (n_samples,)
    '''
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    acc = accuracy_score(y_true, y_pred)
    return acc

def getF1_ordinal(y_true, y_pred):
    '''Accuracy metrics
    :param y_true: the ground truth labels, shape: (n_samples,)
    :param y_pred: the predicted score of each class, shape: (n_samples,)
    '''
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    f1 = f1_score(y_true, y_pred, average='macro')
    return f1
