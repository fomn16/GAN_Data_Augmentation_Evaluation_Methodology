from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *
from Modules.Shared.config import *

class TEST(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.FLOWERS
        self.params.datasetNameComplement = 'default'
        
        self.params.nClasses = 5
        self.params.imgChannels = 3
        self.params.imgWidth = 32
        self.params.imgHeight = 32

        self.transformFunction = lambda entry: resizeImg(32, 0, entry)
        self.filterFunction = None

        self.slices = ['train', 'test']