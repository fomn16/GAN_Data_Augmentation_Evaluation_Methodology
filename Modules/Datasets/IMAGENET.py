from Modules.Datasets.Dataset import Dataset
from Modules.Shared.config import *

class IMAGENET(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.IMAGENET
        self.params.datasetNameComplement = 'default'
        
        self.params.nClasses = 1000
        self.params.imgChannels = 3
        self.params.imgWidth = 32
        self.params.imgHeight = 32

        self.transformFunction = None
        self.filterFunction = None

        self.slices = ['train', 'validation']