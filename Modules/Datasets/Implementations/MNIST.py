from Modules.Datasets.Dataset import Dataset
from Modules.Shared.config import *


class MNIST(Dataset):
    def loadParams(self):
        self.params.datasetName = Datasets.MNIST
        self.params.datasetNameComplement = 'default'

        self.params.nClasses = 10
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28

        self.transformFunction = None
        self.filterFunction = None

        self.slices = ['train', 'test']