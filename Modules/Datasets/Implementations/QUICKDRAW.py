from Modules.Datasets.Dataset import Dataset
from Modules.Shared.helper import *
from Modules.Shared.config import *

class QUICKDRAW(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.QUICKDRAW
        self.params.datasetNameComplement = 'default'
        
        self.params.nClasses = 345
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28

        self.transformFunction = None
        self.filterFunction = None

        self.slices = ['train', 'test']