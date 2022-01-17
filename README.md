# RetinaMNIST: Research
### Based on Project ([Website](https://medmnist.github.io/)) | Dataset ([Zenodo](https://doi.org/10.5281/zenodo.5208230)) | Paper ([arXiv](https://arxiv.org/abs/2110.14795)) | MedMNIST v1 ([ISBI'21](https://medmnist.github.io/v1)) 
[Jiancheng Yang](https://jiancheng-yang.com/), Rui Shi, [Donglai Wei](https://donglaiw.github.io/), Zequan Liu, Lin Zhao, [Bilian Ke](https://scholar.google.com/citations?user=2cX5y8kAAAAJ&hl=en), [Hanspeter Pfister](https://scholar.google.com/citations?user=VWX-GMAAAAAJ&hl=en), [Bingbing Ni](https://scholar.google.com/citations?user=eUbmKwYAAAAJ)

## Introduction
In my research I tried improve baseline metrics described in the original paper and compare some methods for Ordinal Regression task. 
Ordinal regression is a common task in medicine, it is often necessary to grade some image on an ordinal scale (for example, from 0 to 4, where 0 is the absence of a disease, 4 is its strongest form)

This task possible to solve as a Multyclass Classification where each example this is different class-label and training the classifier with softmax activation. Also we can consider this task as a Regression with lenear activation, or we can normalize ordinal scale from 0 to 1, where 0 - this is absence of a disease and 1 and training the Regression with sigmoid activation.

Results of google research:
- [Simple introduction](http://fa.bianp.net/blog/2013/logistic-ordinal-regression/) to classic task of the Ordinal Regression
- [Age Estimation](https://paperswithcode.com/task/age-estimation) is the task of estimating the age of a person from an image and this is also part of the Ordinal regression. So, methods for predict the age of the humaps also compare with methods for us and we can use it. Most popular [repo](https://github.com/Raschka-research-group/coral-cnn)
- So six years ago was a similar competition on [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/overview)

The autors of the original paper offer us solve this task as a Classifier and I try to improve his methods. Classificator looks good as a baseline which possible to compare with Age Estimation model adapted for our task. 

In this research, __I concentrate on improving the baseline__ and I will suppose that __my next step will be training the Age Estimator__.

## Instalation
- install Python 3.8.10
- run ```pip install -r requirements.txt```

## TL;DR
During this work quality metrics was be improved from baseline metrics.

__ACC: 0.513 -> 0.533; AUC: 0.712 -> 0.753__

For reproduction this results run (result of training will be save to ./logs):

```python main.py --augment True --loss focal_loss --dropout 0.4```

## Baseline from paper
I try to repeat the results of autors of the MedMNIST on Resnet-18 (28) model from his [paper](https://arxiv.org/abs/2110.14795).

I implement ResNets with a simple early-stopping strategy on validation set as baseline methods. The input channel is always 3 since we convert gray-scale images into RGB images. To fairly compare with other methods, the input resolutions is 28 for the ResNet-18. I use cross entropy-loss and set the batch size as 128. I utilize an Adam optimizer with an initial learning rate of 0.001 and train the model for 100 epochs, delaying the learning rate by 0.1 after 50 and 75 epochs.

Metrics:
- Val ACC: 0.483, Val AUC: 0.780
- Test ACC: 0.513, Test AUC: 0.712

For reproduction this results run (result of training will be save to ./logs):

```python main.py```

## Add new features to training process
I implement some features which use for training models on the dataset with sparse destribution:
- Image augmentations always helpful for any tasks
- Impliment Focal Loss as a obviusly loss for dataset with sparse destribution
- Replace default Relu activation in the fully-connected classifier to Silu
- Add drpout to fully-connected classifier
- Use Gaussian blur as a Label Smoothing, because I suppose that neighboring labels can have a similar features and hard for separation
- Also I implement classic Label Smoothing, this is powerfull method for training classifier
- Try some optimizers
- Try some learning rate shedulers
- Add Cutmix augmentation
- Add Mixup augmentation

### Add base augmentations
I add base image augmentations as a Transpose, Flip, Rotate, Scale, RandomBrightnes and Contrast, Hue Saturation and Grid Distortion. Augmentation give us big improve for metrics.
I set up propability for every augmentation to 0.5.

Metrics:
- Val ACC: 0.592, Val AUC: 0.835
- Test ACC: 0.522, Test AUC: 0.738

__ACC: 0.513 -> 0.522; AUC: 0.712 -> 0.738__

```python main.py --augment True```

### Add Focal Loss
I use Weighted Focal Loss where alpha coeficient of Focal Loss this is invert labels destribultion. And gamma equal 3 - this got the best results. 

Metrics:
- Val ACC: 0.566, Val AUC: 0.835
- Test ACC: 0.530, Test AUC: 0.746

__ACC: 0.522 -> 0.530; AUC: 0.738 -> 0.746__

```python main.py --augment True --loss focal_loss```

### Replace RELU activation to SiLU
Replace Relu activation with Silu. On many benchmarks, Silu looks better than Relu.

Metrics:
- Val ACC: 0.533, Val AUC: 0.824
- Test ACC: 0.528, Test AUC: 0.743

ACC: 0.530 -> 0.528; AUC: 0.746 -> 0.743

```python main.py --augment True --loss focal_loss --activation silu```

### Add Dropout
Add dropout to Fully-Connected classifier for robusting to ower-fitting.
I used dropout propability equal 0.4 - this got the best results.

Metrics:
- Val ACC: 0.608, Val AUC: 0.838
- Test ACC: 0.533, Test AUC: 0.753

__ACC: 0.530 -> 0.533; AUC: 0.746 -> 0.753__

```python main.py --augment True --loss focal_loss --dropout 0.4```

### Use Gaussian blur as a Label Smoothing
I assume that neighboring labels have similar functions and are similar to each other, and if I apply a Gaussian blur to the labels, then the model may be more robust to ower-fitting. In addition, this method can be profitable for us if our labels in the dataset have poor markup quality. 

I use Gauss kernel for 1D Convolution equal [0.05, 0.9, 0.05], it got the best results.

Metrics:
- Val ACC: 0.575, Val AUC: 0.829
- Test ACC: 0.505, Test AUC: 0.738

ACC: 0.533 -> 0.505; AUC: 0.753 -> 0.738

```python main.py --augment True --loss focal_loss --dropout 0.4 --label_smoothing norm```

### Add Classic Label Smoothing
Classic Label Smoothing has the opportunity for improving the quality. But in many cases, it works only for datasets with poorly marked up.
I use smoothing coeficient equal 0.1

Metrics:
- Val ACC: 0.575, Val AUC: 0.847
- Test ACC: 0.522, Test AUC: 0.749

ACC: 0.533 -> 0.522; AUC: 0.753 -> 0.749

```python main.py --augment True --loss focal_loss --dropout 0.4 --label_smoothing classic```

### Try some Optimizers
I tried 'Adam', 'AdamW', 'RMSprop', but default Adam optimizers got the best results.

### Try some Scheduler 
I tried 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR' and the best result achive for 'CosineAnnealingLR'.

Metrics:
- Val ACC: 0.633, Val AUC: 0.852
- Test ACC: 0.543, Test AUC: 0.745

__ACC: 0.533 -> 0.543__; AUC: 0.753 -> 0.745

```python main.py --augment True --loss focal_loss --dropout 0.4 --scheduler CosineAnnealingLR```

### Cutmix augmentation
It is unlikely that Cutmix will be able to improve the quality of the model, since such augmentation is obviously not effective for this dataset. But I'm still thinking that I need to check it.

Metrics:
- Val ACC: 0.550, Val AUC: 0.816
- Test ACC: 0.505, Test AUC: 0.718

ACC: 0.533 -> 0.505; AUC: 0.753 -> 0.718

```python main.py --augment True --loss focal_loss --dropout 0.4 --cutmix_rate 0.5```

### Mixup augmentation
I think, that Maxup also decreases the quality of the model, but I will check it.

Metrics:
- Val ACC: 0.516, Val AUC: 0.820
- Test ACC: 0.483, Test AUC: 0.726

ACC: 0.533 -> 0.483; AUC: 0.753 -> 0.726

```python main.py --augment True --loss focal_loss --dropout 0.4 --mixup_rate 0.5```

## Summary

Total improve metrics: __ACC: 0.513 -> 0.533; AUC: 0.712 -> 0.753__

| Method               |  AUC  |  ACC  |
|----------------------|:-----:|------:|
| ResNet-18 (28)       | 0.717 | 0.524 |
| ResNet-18 (224)      | 0.710 | 0.493 |
| ResNet-50 (28)       | 0.726 | 0.528 |
| ResNet-50 (224)      | 0.716 | 0.511 |
| auto-sklearn         | 0.690 | 0.515 |
| AutoKeras            | 0.719 | 0.503 |
| Google AutoML Vision | 0.750 | 0.531 |
|----------------------|:-----:|------:|
| ResNet-18 (28)       | 0.712 | 0.513 |
| __ResNet-18 (28)+__  | __0.753__ | __0.533__ |

## Next Steps

- First, need to implement a solution for Age Estimation for our task because they are similar. [This repo looks good.](https://github.com/Raschka-research-group/coral-cnn)
- The second, augmentation is the most important in this task. We can use [optuna](https://optuna.org) or [AutoAlbument](https://albumentations.ai/docs/autoalbument/) for finding the best augmentations.
- Third, we also can use [optuna](https://optuna.org) for finding hyperparameters.
- And the four, of course, we can try some Top-10 methods from [this competition.](https://www.kaggle.com/c/diabetic-retinopathy-detection/overview)