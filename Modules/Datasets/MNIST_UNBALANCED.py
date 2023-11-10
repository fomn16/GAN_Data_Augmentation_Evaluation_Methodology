from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *

class MNIST_UNBALANCED(Dataset):
    def loadParams(self):
        self.params.datasetName = 'mnist'
        self.params.datasetNameComplement = 'unbalanced'

        self.params.nClasses = 10
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28 

        self.transformFunction = None
        self.filterFunction = lambda img, data: unbalance(img, data, 250, self.params.nClasses)
        
        self.slices = ['train', 'test']