import sys
sys.path.insert(1, '../..')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Shared.config import *

#imgs => [w][h][-1,1], lbls => [0,1]
class Dataset:
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()
        self.name = params.datasetName + "_" + self.params.datasetNameComplement
    
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
    
    def getTrainData(self, start, end):
        return self.trainImgs[start:end], self.trainLbls[start:end]

    def getTestData(self, start, end):
        return self.testImgs[start:end], self.testLbls[start:end]
    
    def load(self):
        self.loadParams()
        imgs,lbls = LoadDataset(
                        self.params.datasetName, 
                        True, 
                        True, 
                        self.params.dataDir, 
                        self.params.datasetNameComplement, 
                        self.slices,
                        self.transformFunction,
                        self.filterFunction
                    )

        #número total de instâncias
        self.totalInstances = lbls.shape[0]
        #número de instâncias em cada divisão do fold que vai para treinamento
        self.n_instances_fold_train = int(np.floor((self.totalInstances/self.params.kFold)))
        #numero de instâncias de treinamento nesse fold
        self.trainInstances = self.n_instances_fold_train*(self.params.kFold - 1)
        #numero de instâncias de teste nesse fold
        self.testInstances = self.totalInstances - self.trainInstances

        testStart = self.params.currentFold*self.n_instances_fold_train
        testEnd = testStart + self.testInstances
        self.testImgs = imgs[testStart:testEnd]
        self.testLbls = lbls[testStart:testEnd]
        self.trainImgs = np.concatenate((imgs[:testStart],imgs[testEnd:]))
        self.trainLbls = np.concatenate((lbls[:testStart],lbls[testEnd:]))

    def unload(self):
        del self.testImgs, self.testLbls, self.trainImgs, self.trainLbls