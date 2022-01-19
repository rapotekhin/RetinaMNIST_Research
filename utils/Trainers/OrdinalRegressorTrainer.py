import os, random, gc
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from models.Resnet18 import resnet18
from utils.Datasets.RetinaMnistDatasetOrdinalRegression import RetinaMnistDatasetOrdinalRegression
from utils.Trainers.SimpleTrainer import SimpleTrainer
from scripts.metrics import getACC_ordinal, getF1_ordinal

class OrdinalRegressorTrainer(SimpleTrainer):
    """
    Summary: OrdinalRegressorTrainer
    References:
        https://github.com/Raschka-research-group/coral-cnn/blob/master/model-code/cacd-ordinal.py
    """
    def __init__(self, args: dict) -> None:
        """
        Summary: Initialization.
        Parameters:
            args: dict - config of training. See ./main.py.
        """
        super(OrdinalRegressorTrainer).__init__()
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imp = torch.ones(self.args['nb_classes']-1, dtype=torch.float)
        self._seed_torch()
    
    
    def run(self) -> None:
        """
        Summary: Running the training process.
        """
        
        # load the model
        model = resnet18(self.args['nb_classes'], False)

        # load the data
        train_dataset = RetinaMnistDatasetOrdinalRegression(split='train', augment=self.args['augment'], download=True, args=self.args)
        val_dataset = RetinaMnistDatasetOrdinalRegression(split='val', augment=self.args['augment'], download=True, args=self.args)
        test_dataset = RetinaMnistDatasetOrdinalRegression(split='test', augment=self.args['augment'], download=True, args=self.args)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.args['batch_size'], shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=self.args['batch_size'], shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*self.args['batch_size'], shuffle=False)

        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args['lr']) 
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        best_acc, best_f1, best_epoch = -1, -1, -1
        
        for epoch in range(self.args['epochs']):
            running_loss = 0.0
            dataset_size = 0
            model.train()
            for batch_idx, (features, targets, levels) in enumerate(train_loader):

                features = features.to(self.device)
                targets = targets
                targets = targets.to(self.device)
                levels = levels.to(self.device)
                # FORWARD AND BACK PROP
                logits, probas = model(features)

                cost = self.cost_fn(logits, levels, self.imp)
                optimizer.zero_grad()

                cost.backward()

                # UPDATE MODEL PARAMETERS
                optimizer.step()

                running_loss += cost.item() * self.args['batch_size']
                dataset_size += self.args['batch_size'] 
                epoch_loss = running_loss / dataset_size

                # LOGGING
                if not batch_idx % 2:
                    s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                        % (epoch+1, self.args['epochs'], batch_idx,
                            len(train_dataset)//self.args['batch_size'], epoch_loss))
                    print(s)

            scheduler.step()

            model.eval()
            with torch.set_grad_enabled(False):
                valid_acc, valid_f1, val_loss = self.compute_acc_and_f1(model, val_loader)

            if valid_f1 > best_f1:
                best_acc, best_f1, best_epoch = valid_acc, valid_f1, epoch
                ########## SAVE MODEL #############
                torch.save(model.state_dict(), os.path.join(self.args['path_to_save'], 'best_model.pth'))


            s = 'acc/f1: | Current Valid: %.2f/%.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
                valid_acc, valid_f1, val_loss, epoch, best_acc, best_f1, best_epoch)
            print(s)

        model.eval()
        with torch.set_grad_enabled(False):  # save memory during inference

            train_acc, train_f1, _ = self.compute_acc_and_f1(model, train_loader)
            valid_acc, valid_f1, _ = self.compute_acc_and_f1(model, val_loader)
            test_acc, test_f1, _ = self.compute_acc_and_f1(model, test_loader)

            s = 'acc/f1: | Train: %.2f/%.2f | Valid: %.2f/%.2f | Test: %.2f/%.2f' % (
                train_acc, train_f1,
                valid_acc, valid_f1,
                test_acc, test_f1)

        print(s)


        ########## EVALUATE BEST MODEL ######
        model.load_state_dict(torch.load(os.path.join(self.args['path_to_save'], 'best_model.pth')))
        model.eval()

        with torch.set_grad_enabled(False):
            train_acc, train_f1, _ = self.compute_acc_and_f1(model, train_loader)
            valid_acc, valid_f1, _ = self.compute_acc_and_f1(model, val_loader)
            test_acc, test_f1, _ = self.compute_acc_and_f1(model, test_loader)

            s = 'acc/f1: | Best Train: %.2f/%.2f | Best Valid: %.2f/%.2f | Best Test: %.2f/%.2f' % (
                train_acc, train_f1,
                valid_acc, valid_f1,
                test_acc, test_f1)
            print(s)

    def cost_fn(self, logits, levels, imp):
        val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))
        return torch.mean(val)
        
    def compute_acc_and_f1(self, model, data_loader):
        acc, f1, iters = 0, 0, 1
        dataset_size, running_loss = 0, 0
        for i, (features, targets, levels) in enumerate(data_loader):
            
            features = features.to(self.device)
            targets = targets.to(self.device)

            logits, probas = model(features)

            cost = self.cost_fn(logits, levels, self.imp)

            running_loss += cost.item() * self.args['batch_size']
            dataset_size += self.args['batch_size'] 
            epoch_loss = running_loss / dataset_size

            predict_levels = probas > 0.5
            predicted_labels = torch.sum(predict_levels, dim=1)

            iters += i

            targets = targets.cpu().detach().numpy()
            predicted_labels = predicted_labels.cpu().detach().numpy()

            acc += getACC_ordinal(targets, predicted_labels)
            
            f1 += getF1_ordinal(targets, predicted_labels)
            
        acc = acc / iters
        f1 = f1 / iters
        return acc, f1, epoch_loss