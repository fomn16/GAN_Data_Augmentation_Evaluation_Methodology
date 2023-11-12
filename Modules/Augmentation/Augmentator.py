import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

from Modules.Datasets.Dataset import Dataset

#imgs => [w][h][-1,1], lbls => [0,1]
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