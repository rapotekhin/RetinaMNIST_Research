
import timm
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, args: dict, pretrained: bool=True):
        super().__init__()
        self.activation = nn.SiLU() if args['activation'] == 'silu' else nn.ReLU()
        self.backbone = timm.create_model(args['model_name'], pretrained=pretrained, in_chans=3)
        self.backbone.fc = nn.Sequential(nn.Linear(self.backbone.fc.in_features, 128),
                                         self.activation,
                                         nn.Dropout(p=args['dropout']),
                                         nn.Linear(128, args['nb_classes']))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
        return x