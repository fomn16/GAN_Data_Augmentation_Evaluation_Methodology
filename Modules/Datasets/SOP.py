import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Datasets.Dataset import Dataset

class STANFORD_ONLINE_PRODUCTS(Dataset):
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()

        self.name = params.datasetName
        self.dataset, self.info = tfds.load(name = params.datasetName, with_info=True, data_dir=params.dataDir)

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
    
    def getTrainData(self, start, end):
        return self._getFromDatasetLL(start, end)

    def getTestData(self, start, end):
        return self._getFromDatasetLL(start, end, True)
    
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
    def _getFromDatasetLL(self, start, end, test=False):
        if(test):
            start += self.params.currentFold*self.n_instances_fold
            end += self.params.currentFold*self.n_instances_fold
        imgs = None
        lbls = None
        if(end < self.trainInstances):
            imgs, lbls = loadIntoArrayLL('train', self.dataset, self.params.nClasses, start, end, 'image', 'label', self.resizeImg)
        elif (start >= self.trainInstances):
            imgs, lbls = loadIntoArrayLL('test', self.dataset, self.params.nClasses, start - self.trainInstances, end - self.trainInstances, 'image', 'label')
        else:
            imgs1, lbls1 = loadIntoArrayLL('train', self.dataset, self.params.nClasses, start, self.trainInstances - 1, 'image', 'label')
            imgs2, lbls2 = loadIntoArrayLL('test', self.dataset, self.params.nClasses, 0, end - self.trainInstances, 'image', 'label')
            imgs = np.concatenate((imgs1, imgs2))
            lbls = np.concatenate((lbls1, lbls2))
            del imgs1, imgs2, lbls1, lbls2
        return imgs, lbls