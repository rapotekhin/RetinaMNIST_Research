from abc import ABC
import warnings
warnings.filterwarnings("ignore") 


class AbstractEvaluator(ABC):

    def __init__(self):
        pass

    def run(self):
        pass