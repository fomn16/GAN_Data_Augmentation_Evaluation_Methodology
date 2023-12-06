from Modules.Datasets.Dataset import Dataset

from Modules.Shared.config import *

class CIFAR_10(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.CIFAR_10
        self.params.datasetNameComplement = 'default'
        
        self.params.nClasses = 10
        self.params.imgChannels = 3
        self.params.imgWidth = 32
        self.params.imgHeight = 32

        self.transformFunction = None
        self.filterFunction = None

        self.slices = ['train', 'test']