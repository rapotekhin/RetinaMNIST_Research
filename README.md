# RetinaMNIST_Research
This is simple research of the RetinaMNIST dataset

# Introduction
Introduction

# TL;DR
TL;DR

# Baseline from paper
Resnet18 (28x28):
- val_best_acc 0.592, val_best_auc 0.789
- test_best_acc 0.518, test_best_auc 0.720

```python main.py```

# Add new features to training process
## Add base augmentations
Resnet18 (28x28):
- val_best_acc 0.591, val_best_auc 0.842
- test_best_acc 0.538, test_best_auc 0.754

__ACC: 0.518 -> 0.538; AUC: 0.720 -> 0.754__

```python main.py --augment True```

## Add Focal Loss
gamma = 3, alpha = inverse destribution
Resnet18 (28x28):
- val_best_acc 0.583, val_best_auc 0.812
- test_best_acc 0.540, test_best_auc 0.759

__ACC: 0.537 -> 0.540; AUC: 0.753 -> 0.759__

```python main.py --augment True --loss focal_loss```

## Replace RELU activation to SiLU
Resnet18 (28x28):
- val_best_acc 0.558, val_best_auc 0.803
- test_best_acc 0.525, test_best_auc 0.735

ACC: 0.540 -> 0.525; AUC: 0.759 -> 0.735

```python main.py --augment True --loss focal_loss --activation silu```

## Add Dropout
best propability = 0.05
Resnet18 (28x28):
- val_best_acc 0.625, val_best_auc 0.828
- test_best_acc 0.530, test_best_auc 0.733

ACC: 0.540 -> 0.530; AUC: 0.759 -> 0.733

```python main.py --augment True --loss focal_loss --dropout 0.05```

## Add Gaussian Label Smoothing
for best gauss kernel = [0.05, 0.9, 0.05]
Resnet18 (28x28):
- val_best_acc 0.608, val_best_auc 0.851
- test_best_acc 0.5275, test_best_auc 0.736

ACC: 0.540 -> 0.527; AUC: 0.759 -> 0.736

```python main.py --augment True --loss focal_loss --label_smoothing norm```

## Add Classic Label Smoothing
for best smoophing = 0.1
Resnet18 (28x28):
- val_best_acc 0.575, val_best_auc 0.824
- test_best_acc 0.525, test_best_auc 0.742

ACC: 0.540 -> 0.525; AUC: 0.759 -> 0.742

```python main.py --augment True --loss focal_loss --label_smoothing classic```

## Choising Optimizer
I tried 'Adam', 'AdamW', 'RMSprop'
default Adam the best result

## Choising Scheduler
I tried 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR' and the best result achive for 'ExponentialLR'
Resnet18 (28x28):
- val_best_acc 0.608, val_best_auc 0.827
- test_best_acc 0.545, test_best_auc 0.759

__ACC: 0.540 -> 0.545__; AUC: 0.759 -> 0.759


# Training ResNet-50 (28) with best parameters
Resnet50 (28x28):
- val_best_acc 0.567, val_best_auc 0.788
- test_best_acc 0.512, test_best_auc 0.723

ACC: 0.545 -> 0.512; AUC: 0.759 -> 0.723

```python main.py --augment True --loss focal_loss --scheduler ExponentialLR --model_name resnet50```

# Summary
Total improve metrics: __ACC: 0.517 -> 0.545; AUC: 0.720 -> 0.759__

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
| ResNet-18 (28)       | 0.720 | 0.518 |
| __ResNet-18 (28)+__  | __0.759__ | __0.545__ |
| ResNet-50 (28)+      | 0.723 | 0.512 |

# Next steps and Roadmap
Next steps and Roadmap