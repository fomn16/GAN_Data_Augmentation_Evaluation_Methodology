from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *
from Modules.Shared.config import *

class EUROSAT(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.TEST
        self.params.datasetNameComplement = 'default'
        
        self.params.nClasses = 10
        self.params.imgChannels = 3
        self.params.imgWidth = 48
        self.params.imgHeight = 48

        self.transformFunction = lambda entry: resizeImg(48, 0, entry)
        self.filterFunction = None

        self.slices = ['train']