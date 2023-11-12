from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *

class CIFAR_10_UNBALANCED(Dataset):
    def loadParams(self):
        self.params.datasetName = 'cifar10'
        self.params.datasetNameComplement = 'unbalanced'
        
        self.params.nClasses = 10
        self.params.imgChannels = 3
        self.params.imgWidth = 32
        self.params.imgHeight = 32

        self.transformFunction = None
        self.filterFunction = lambda img, data: unbalance(img, data, 250, self.params.nClasses)

        self.slices = ['train', 'test']