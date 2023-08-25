import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Datasets.Dataset import Dataset

class CIFAR_10(Dataset):
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()

        self.name = params.datasetName
        self.dataset, self.info = tfds.load(name = params.datasetName, with_info=True, data_dir=params.dataDir)

        #numero de instancias nos splits de treinamento e teste no dataset original
        self.trainInstancesDataset = self.info.splits['train'].num_examples
        self.testInstancesDataset = self.info.splits['test'].num_examples
        #número total de instâncias
        self.totalInstances = int(np.floor((self.trainInstancesDataset + self.testInstancesDataset)))
        #número de instâncias em cada divisão do fold que vai para treinamento
        self.n_instances_fold_train = int(np.floor(self.totalInstances/self.params.kFold))
        #numero de instâncias de treinamento nesse fold
        self.trainInstances = self.n_instances_fold_train*(self.params.kFold - 1)
        #numero de instâncias de teste nesse fold
        self.testInstances = self.totalInstances - self.trainInstances

        self.trainDataset = getFromDatasetLL(0, self.trainInstances, params.currentFold, self.n_instances_fold_train,
                                             self.testInstances, self.trainInstancesDataset, params.nClasses,
                                             self.dataset, 'image', 'label')
        
        self.testDataset = getFromDatasetLL(0, self.testInstances, params.currentFold, self.n_instances_fold_train,
                                             self.testInstances, self.trainInstancesDataset, params.nClasses,
                                             self.dataset, 'image', 'label', test=True)
    
    def loadParams(self):
        self.params.datasetName = 'cifar10'
        self.params.nClasses = 10
        self.params.imgChannels = 3
        self.params.imgWidth = 32
        self.params.imgHeight = 32
    
    def getTrainData(self, start, end):
        imgs, lbls = self.trainDataset
        return imgs[start:end], lbls[start:end]

    def getTestData(self, start, end):
        imgs, lbls = self.testDataset
        return imgs[start:end], lbls[start:end]