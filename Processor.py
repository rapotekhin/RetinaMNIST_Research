import os, json
from datetime import datetime

from utils.Trainers.SimpleTrainer import SimpleTrainer

class Processor:

    def __init__(self, args: dict) -> None:
        """
        Summary: Initialization.
        Parameters:
            args: dict - config of training. See ./main.py.
        """
        self.args = args
        self._set_datetime_to_config()

        self.pytorch_simple_trainer = SimpleTrainer(self.args)

    def run(self) -> None:
        """
        Summary: Running the module
        """
        if self.args['mode'] == 'train':
            print("BEGIN TRAINING")
            self._train()
            self._save_config()
        else:
            raise ValueError('Unknown mode: {}'.format(self.args['mode']))


    def _train(self) -> None:
        """
        Summary: Trainin the model with user parameters 
        """
        self.pytorch_simple_trainer.run()

    
    def _save_config(self) -> None:
        """
        Summary: Save the config file
        """
        temp_config = self.args
        with open(os.path.join(self.args['path_to_save'], 'config.json'), 'w') as f:
            json.dump(temp_config, f, indent=4)

        del temp_config

    def _set_datetime_to_config(self) -> None:
        """
        Summary: Get the current date and time and create directory for saving results
        """
        self.args['datetime_start'] = datetime.now().strftime("%d%m%Y_%H%M%S")

        folder_name = '{}-{}-{}'.format(self.args['datetime_start'], 
                                        self.args['model_name'], 
                                        self.args['mode'])

        self.args['path_to_save'] = os.path.join(self.args['path_to_save'], folder_name)
        os.makedirs(self.args['path_to_save'], exist_ok=False)