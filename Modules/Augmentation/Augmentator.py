import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

from Modules.Datasets.Dataset import Dataset
from Modules.Shared.Saving import *

class Augmentator:
    name = None
    def __init__(self, params:Params, extraParams = None, nameComplement = ""):
        pass

    def compile(self):
        pass

    def train(self, dataset: Dataset):
        pass

    def saveGenerationExample(self, nEntries = 20):
        pass

    def generate(self, srcImgs, srcLbls):
        pass
    
    def verifyInitialization(self, dataset: Dataset):
        if(self.generator is None):
            if(loadParam(self.name + '_current_epoch', 0) == 0):
                saveParam(self.name + '_current_epoch', self.ganEpochs-1)
                self.params.continuing = True
            self.compile()