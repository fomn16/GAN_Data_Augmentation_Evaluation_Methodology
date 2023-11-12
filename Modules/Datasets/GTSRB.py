from Modules.Datasets.Dataset import Dataset


class GTSRB(Dataset):
    def loadParams(self):
        self.params.datasetName = 'quickdraw_bitmap'
        self.params.datasetNameComplement = 'default'
        
        self.params.nClasses = 345
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28

        self.transformFunction = None
        self.filterFunction = None

        self.slices = ['train', 'test']