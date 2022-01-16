from typing import Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import gc
import os, random
from PIL import Image

import timm

import medmnist
from medmnist import INFO, Evaluator
from medmnist.dataset import MedMNIST2D

from models.RetinaMNISTModel import Net
from utils.Datasets.RetinaMNISTDataset import RetinaMNISTDataset
from utils.Trainers.Abstract import AbstractTrainer
from utils.Losses.WeightedFocalLoss import WeightedFocalLoss

class SimpleTrainer(AbstractTrainer):

    def __init__(self, args):
        super(SimpleTrainer).__init__()
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._seed_torch()

    def run(self):
        
        # load the model
        net = Net(self.args)

        # load the data
        train_dataset = RetinaMNISTDataset(split='train', augment=self.args['augment'], download=True, args=self.args)
        val_dataset = RetinaMNISTDataset(split='val', augment=self.args['augment'], download=True, args=self.args)
        test_dataset = RetinaMNISTDataset(split='test', augment=self.args['augment'], download=True, args=self.args)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.args['batch_size'], shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=self.args['batch_size'], shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*self.args['batch_size'], shuffle=False)

        if self.args['loss'] == "focal_loss":
            criterion = WeightedFocalLoss(1, 2)
        elif self.args['loss'] == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError('Unknown loss: {}'.format(self.args['loss']))

        if self.args['optimizer'] == "Adam":
            optimizer = optim.Adam(net.parameters(), lr=self.args['lr'])
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.args['optimizer']))

        if self.args['scheduler'] == "MultiStepLR":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)
        else:
            raise ValueError('Unknown scheduler: {}'.format(self.args['scheduler']))

        self._train(net,
                    train_loader, 
                    val_loader,
                    criterion,
                    optimizer,
                    scheduler
                )

        checkpoint = torch.load('model_best.pth')
        net.load_state_dict(checkpoint['model'])

        val_best_acc, val_best_auc = self._eval(net, val_loader, split='val')
        test_best_acc, test_best_auc = self._eval(net, test_loader, split='test')

        print(f"val_best_acc {val_best_acc}, val_best_auc {val_best_auc}")
        print(f"test_best_acc {test_best_acc}, test_best_auc {test_best_auc}\n")

    def _train(self, net, train_loader, val_loader, criterion, optimizer, scheduler):
        # train
        net.to(self.device)
        
        best_acc, best_auc = 0, 0
        dataset_size = 0
        running_loss = 0.0

        for epoch in range(self.args['epochs']):

            net.train()

            bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")
            for _, (inputs, targets) in bar:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = net(inputs)
                
                targets = targets.squeeze().to(torch.int64)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * self.args['batch_size']
                dataset_size += self.args['batch_size'] 
                epoch_loss = running_loss / dataset_size

                bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

            scheduler.step()
            gc.collect()

            val_acc, val_auc = self._eval(net, val_loader, 'val')

            if val_auc >= best_auc:
                print(f"Validation AUC Improved ({best_auc} ---> {val_auc}), Accuracy: {val_acc}")
                best_auc = val_auc
                best_acc = val_acc 

                torch.save({'model': net.state_dict()}, 'model_best.pth')

        return best_acc, best_auc

    def _eval(self, net, data_loader, split='train') -> tuple((float, float)):

        net.eval()

        y_true = torch.tensor([])
        y_score = torch.tensor([])
        
        with torch.no_grad():

            bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Eval")
            for _, (inputs, targets) in bar:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = net(inputs)
                targets = targets.squeeze().long()
        
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)
                
                targets = targets.cpu()
                outputs = outputs.cpu()
        
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()
            
            evaluator = Evaluator('retinamnist', split)
            metrics = evaluator.evaluate(y_score)

            acc = metrics.ACC
            auc = metrics.AUC

            # bar.set_postfix(Epoch=epoch+1, Valid_Loss=epoch_loss, Valid_ACC=acc, Valid_AUC=auc)   

        return acc, auc

    def _seed_torch(self, seed: int = 42) -> None:
        """
        Summary: Set random seed everywhere.
        Parameters:
            seed: int - random seed
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True