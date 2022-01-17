from utils.Trainers.SimpleTrainer import SimpleTrainer

import json
import os, sys
from datetime import datetime


class Processor:

    def __init__(self, args: dict):
        self.args = args


        self.pytorch_simple_trainer = SimpleTrainer(self.args)
        # self.pytorch_simple_evaluator = SimpleEvaluator(self.args)


    def run(self):
        if self.args['mode'] == 'train':
            print("BEGIN TRAINING")
            self._train()
        elif self.args['mode'] == 'evaluate':
            print("BEGIN EVALUATION")
            self._evaluate()
        else:
            raise ValueError('Unknown mode: {}'.format(self.args['mode']))


    def _train(self):
        """
        Summary: Trainin the model with user parameters 
        """
        self.pytorch_simple_trainer.run()


    def _evaluate(self):
        """
        Summary: Clssic evaluate agorithm, we calculate main metrics and confussion matrics on the test dataset
                 Don't use if you use Meta-Learning already! Because you evaluation results will be wrong.
        """
        self.pytorch_simple_evaluator.run()
