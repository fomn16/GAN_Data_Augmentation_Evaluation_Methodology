import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator

class Benchmark:
    def __init__(self, params: Params, nameComplement = ""):
        pass

    def train(self, augmentator: Augmentator, dataset: Dataset, extraEpochs = 1):
        pass

    def runTest(self, dataset: Dataset):
        pass