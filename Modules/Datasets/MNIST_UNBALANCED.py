from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *
from Modules.Shared.config import *

class MNIST_UNBALANCED(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.MNIST
        self.params.datasetNameComplement = 'unbalanced'

        self.params.nClasses = 10
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28 

        self.transformFunction = None
        self.filterFunction = lambda img, data: unbalance(img, data, 700, self.params.nClasses)
        
        self.slices = ['train', 'test']