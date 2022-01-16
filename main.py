import argparse
from distutils.util import strtobool

from Processor import Processor

def main(args: dict):
    processor = Processor(args)
    processor.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REtinaMNIST: research')

    parser.add_argument('--mode', help='', choices=['train',
                                                    'evaluate'],
                                                    default='train')
    parser.add_argument('--path_to_save', dest='path_to_save', type=str,
                        help='path to save models', default='./logs')
    parser.add_argument('--model_name', dest='model_name', type=str,
                        help='select model from TIMM Hub', choices=['resnet18'], default='resnet18')
    parser.add_argument('--img_size', dest='img_size', type=int, help='img_size', choices=[28, 224], default=28)
    parser.add_argument('--epochs', dest='epochs', type=int, help='epochs', default=5)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--nb_classes', dest='nb_classes', type=int, help='nb_classes', default=5)
    parser.add_argument('--augment', dest='augment', type=lambda x: bool(strtobool(x)), help='augment', default=False)

    parser.add_argument('--loss', dest='loss', type=str,
                        help='Loss function', choices=['focal_loss', 'cross_entropy'], default='cross_entropy')
    parser.add_argument('--lr', dest='lr', type=float, help='lr', default=0.001)                
    parser.add_argument('--scheduler', dest='scheduler', type=str,
                        help='lr scheduler, MultiStepLR - default in the paper', choices=['MultiStepLR'], default='MultiStepLR')
    parser.add_argument('--optimizer', dest='optimizer', type=str,
                        help='optimizer', choices=['Adam'], default='Adam')

    # Augmentations
    parser.add_argument('--Transpose', dest='Transpose', type=float, help='Transpose', default=0.5)
    parser.add_argument('--HorizontalFlip', dest='HorizontalFlip', type=float, help='HorizontalFlip', default=0.5)
    parser.add_argument('--VerticalFlip', dest='VerticalFlip', type=float, help='VerticalFlip', default=0.5)
    parser.add_argument('--GridDistortion', dest='GridDistortion', type=float, help='GridDistortion', default=0.5)
    parser.add_argument('--ShiftScaleRotate', dest='ShiftScaleRotate', type=float, help='ShiftScaleRotate', default=0.5)
    parser.add_argument('--HueSaturationValue', dest='HueSaturationValue', type=float, help='HueSaturationValue', default=0.5)
    parser.add_argument('--RandomBrightnessContrast', dest='RandomBrightnessContrast', type=float, help='RandomBrightnessContrast', default=0.5)

    args = vars(parser.parse_args())
    main(args)