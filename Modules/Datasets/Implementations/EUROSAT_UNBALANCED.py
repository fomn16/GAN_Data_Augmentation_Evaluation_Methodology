from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *
from Modules.Shared.config import *

class EUROSAT_UNBALANCED(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.TEST
        self.params.datasetNameComplement = 'unbalanced'
        
        self.params.nClasses = 10
        self.params.imgChannels = 3
        self.params.imgWidth = 48
        self.params.imgHeight = 48

        self.transformFunction = lambda entry: resizeImg(48, 0, entry)
        self.filterFunction = lambda img, data: unbalance(img, data, 0.1, self.params.nClasses)

        self.slices = ['train']