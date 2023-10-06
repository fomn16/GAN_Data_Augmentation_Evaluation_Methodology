import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Datasets.Dataset import Dataset
class STANFORD_ONLINE_PRODUCTS(Dataset):
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()

        self.lastStart = None
        self.lastEnd = None
        self.lastQuery = None
        self.maxBatchesInMemory = 250
        self.data = None

        self.name = self.params.datasetName
        self.dataset, self.info = tfds.load(name = self.params.datasetName, with_info=True, data_dir=self.params.dataDir)
        
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
    
    def loadParams(self):
        self.params.datasetName = 'stanford_online_products'
        self.params.nClasses = 12
        self.params.imgChannels = 3
        self.params.imgWidth = 64
        self.params.imgHeight = 64
    
    def _getDataCache(self, start, end, test):
        if(self.lastStart is None or self.lastEnd is None or self.lastQuery != test or start < self.lastStart or end < self.lastStart or start > self.lastEnd or end > self.lastEnd):
            print('\nloading new dataset cache\n')
            bs = end - start
            fullEnd = start + bs*self.maxBatchesInMemory
            self.data = getFromDatasetLL(start, fullEnd, self.params.currentFold, self.n_instances_fold_train, 
                                         self.testInstances, self.trainInstancesDataset, self.params.nClasses, 
                                         self.dataset, 'image', 'super_class_id', self.resizeImg, test)
            self.lastStart = start
            self.lastEnd = fullEnd
            self.lastQuery = test
            print('\nfinished loading new dataset cache\n')
        
        imgs, lbls = self.data
        start = start - self.lastStart
        end = end - self.lastStart

        return imgs[start:end], lbls[start:end]

    def getTrainData(self, start, end):
        return self._getDataCache(start, end, False)

    def getTestData(self, start, end):
        return self._getDataCache(start, end, True)
    
    def resizeImg(self, entry):
        img = entry['image']
        w = img.shape[0]
        h = img.shape[1]
        side = np.min([w,h])
        cw, ch = int(w/2 - side/2), int(h/2 - side/2)
        img = np.asarray(PIL.Image.fromarray(img[cw:cw+side, ch:ch+side]).resize(size=(self.params.imgWidth,self.params.imgHeight)))
        entry['image'] = img
        return entry