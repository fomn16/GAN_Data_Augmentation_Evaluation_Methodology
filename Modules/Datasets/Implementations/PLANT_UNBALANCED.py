from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *
from Modules.Shared.config import *

class PLANT_UNBALANCED(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.PLANT
        self.params.datasetNameComplement = 'unbalanced'
        
        self.params.nClasses = 38
        self.params.imgChannels = 3
        self.params.imgWidth = 32
        self.params.imgHeight = 32

        self.transformFunction = lambda entry: resizeImg(32, 0, entry)
        self.filterFunction = lambda img, data: unbalance(img, data, 0.1, self.params.nClasses)

        self.slices = ['train']