import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Datasets.Dataset import Dataset


'''6.02%loading new dataset cache
94.62%loading new dataset cache
Traceback (most recent call last):
  File "Z:\felip\Documents\UNB\TCC\modulos\main.py", line 44, in <module>
    augmentator.train(dataset)
  File "Z:\felip\Documents\UNB\TCC\modulos\Modules\Augmentation\SOP\GAN_SOP.py", line 167, in train
    imgBatch, labelBatch = dataset.getTrainData(i*self.batchSize, (i+1)*self.batchSize)
  File "Z:\felip\Documents\UNB\TCC\modulos\Modules\Datasets\SOP.py", line 50, in getTrainData
    def getTrainData(self, start, end):
  File "Z:\felip\Documents\UNB\TCC\modulos\Modules\Datasets\SOP.py", line 38, in getDataCache
    fullEnd = start + bs*self.maxBatchesInMemory
  File "Z:\felip\Documents\UNB\TCC\modulos\Modules\Datasets\SOP.py", line 83, in _getFromDatasetLL
    imgs, lbls = loadIntoArrayLL('train', self.dataset, self.params.nClasses, start, end, 'image', 'super_class_id', self.resizeImg)
  File "Z:\felip\Documents\UNB\TCC\modulos\Modules\Shared\helper.py", line 93, in loadIntoArrayLL
    imgs = np.zeros((end - start,) + output_shapes).astype('float')
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 8.66 GiB for an array with shape (3230, 400, 300, 3) and data type float64'''
class STANFORD_ONLINE_PRODUCTS(Dataset):
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()

        self.name = params.datasetName
        self.dataset, self.info = tfds.load(name = params.datasetName, with_info=True, data_dir=params.dataDir)
        
        #corrigir para que train/test instances sejam as realmente contidas no train e test de acordo com o fold. Usar outra variavel internamente se necessario saber sobre as secções do dataset real
        
        self.trainInstances = self.info.splits['train'].num_examples
        self.testInstances = self.info.splits['test'].num_examples
        self.totalInstances = self.trainInstances + self.testInstances
        self.n_instances_fold = int(np.floor(self.totalInstances/self.params.kFold))
    
    def loadParams(self):
        self.params.datasetName = 'stanford_online_products'
        self.params.nClasses = 12
        self.params.imgChannels = 3
        self.params.imgWidth = 128
        self.params.imgHeight = 128

        self.lastStart = None
        self.lastEnd = None
        self.lastQuery = None
        self.maxBatchesInMemory = 150
        self.data = None
    
    def getDataCache(self, start, end, test):
        if(self.lastStart is None or self.lastEnd is None or self.lastQuery != test or start < self.lastStart or end < self.lastStart or start > self.lastEnd or end > self.lastEnd):
            print('\nloading new dataset cache\n')
            bs = end - start
            fullEnd = start + bs*self.maxBatchesInMemory
            self.data = self._getFromDatasetLL(start, fullEnd, test)
            self.lastStart = start
            self.lastEnd = fullEnd
            self.lastQuery = test
        
        imgs, lbls = self.data
        start = start - self.lastStart
        end = end - self.lastStart

        return imgs[start:end], lbls[start:end]

    def getTrainData(self, start, end):
        return self.getDataCache(start, end, False)

    def getTestData(self, start, end):
        return self.getDataCache(start, end, True)
    

    #corrigir, usar kfold em vez de train/test instances
    def getAllTrainData(self):
        return self._getFromDatasetLL(0, self.trainInstances-1)
    
    def getAllTestData(self):
        return self._getFromDatasetLL(0, self.testInstances-1, True)
    
    def resizeImg(self, entry):
        img = entry['image']
        w = img.shape[0]
        h = img.shape[1]
        side = np.min([w,h])
        cw, ch = int(w/2 - side/2), int(h/2 - side/2)
        img = np.asarray(PIL.Image.fromarray(img[cw:cw+side, ch:ch+side]).resize(size=(self.params.imgWidth,self.params.imgHeight)))
        entry['image'] = img
        return entry

    #carrega instancias do dataset usando lazy loading dos dados
    #corrigir uso de kfold
    def _getFromDatasetLL(self, start, end, test=False):
        if(test):
            start += self.params.currentFold*self.n_instances_fold
            end += self.params.currentFold*self.n_instances_fold
        imgs = None
        lbls = None
        if(end < self.trainInstances):
            imgs, lbls = loadIntoArrayLL('train', self.dataset, self.params.nClasses, start, end, 'image', 'super_class_id', self.resizeImg)
        elif (start >= self.trainInstances):
            imgs, lbls = loadIntoArrayLL('test', self.dataset, self.params.nClasses, start - self.trainInstances, end - self.trainInstances, 'image', 'super_class_id')
        else:
            imgs1, lbls1 = loadIntoArrayLL('train', self.dataset, self.params.nClasses, start, self.trainInstances - 1, 'image', 'super_class_id')
            imgs2, lbls2 = loadIntoArrayLL('test', self.dataset, self.params.nClasses, 0, end - self.trainInstances, 'image', 'super_class_id')
            imgs = np.concatenate((imgs1, imgs2))
            lbls = np.concatenate((lbls1, lbls2))
            del imgs1, imgs2, lbls1, lbls2
        return imgs, lbls