import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Shared.DatasetIteratorInfo import DatasetIteratorInfo

class MNIST:
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()

        self.name = params.datasetName
        self.dataset, self.info = tfds.load(name = params.datasetName, with_info=True, as_supervised=True, data_dir=params.dataDir)

        imgs1, lbls1 = loadIntoArray(self.dataset['train'], params.nClasses)
        imgs2, lbls2 = loadIntoArray(self.dataset['test'], params.nClasses)
        self.imgs = np.concatenate((imgs1, imgs2))
        self.lbls = np.concatenate((lbls1, lbls2))
        totalEntries = self.imgs.shape[0]
        self.n = int(np.floor(totalEntries/params.kFold))


        self.total_instances = self.info.splits['test'].num_examples + self.info.splits['train'].num_examples
        self.n_instances_fold = int(np.floor(self.total_instances/self.params.kFold))

        self.datasetIterators = DatasetIteratorInfo(self.dataset['train'].as_numpy_iterator(), self.dataset['test'].as_numpy_iterator())

    def loadParams(self):
        self.params.datasetName = 'mnist'
        self.params.nClasses = 10
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28

    def getTrainData(self):
        trainImgs = self.imgs[:self.params.currentFold*self.n]
        trainLbls = self.lbls[:self.params.currentFold*self.n]
        trainImgs = np.concatenate((trainImgs, self.imgs[(self.params.currentFold + 1)*self.n:]))
        trainLbls = np.concatenate((trainLbls, self.lbls[(self.params.currentFold + 1)*self.n:]))
        return trainImgs, trainLbls

    def getTestData(self):
        testImgs = self.imgs[self.params.currentFold*self.n:(self.params.currentFold + 1)*self.n]
        testLbls = self.lbls[self.params.currentFold*self.n:(self.params.currentFold + 1)*self.n]
        return testImgs, testLbls
    
    def getAllData(self):
        imgs1, lbls1 = loadIntoArray(self.dataset['train'], self.params.nClasses)
        imgs2, lbls2 = loadIntoArray(self.dataset['test'], self.params.nClasses)
        imgs = np.concatenate((imgs1, imgs2))
        lbls = np.concatenate((lbls1, lbls2))
        return imgs, lbls
    
    def getNTrain(self):
        return self.info.splits['train'].num_examples
    
    def getNTest(self):
        return self.info.splits['test'].num_examples
    
    #carrega instancias do dataset usando lazy loading dos dados
    def getFromDatasetLL(self, start, end, test=False):
        testEntries = self.info.splits['test'].num_examples
        trainEntries = self.info.splits['train'].num_examples
        totalEntries = testEntries + trainEntries
        n = int(np.floor(totalEntries/self.params.kFold))

        if(test):
            start += self.params.currentFold*n
            end += self.params.currentFold*n
        
        imgs = None
        lbls = None
        if(end < trainEntries):
            imgs, lbls = loadIntoArrayLL('train', self.dataset, start, end, self.datasetIterators)
        elif (start >= trainEntries):
            imgs, lbls = loadIntoArrayLL('test', self.dataset, start - trainEntries, end - trainEntries, self.datasetIterators)
        else:
            imgs1, lbls1 = loadIntoArrayLL('train', self.dataset, start, trainEntries - 1, self.datasetIterators)
            imgs2, lbls2 = loadIntoArrayLL('test', self.dataset, 0, end - trainEntries, self.datasetIterators)
            imgs = np.concatenate((imgs1, imgs2))
            lbls = np.concatenate((lbls1, lbls2))
            del imgs1, imgs2, lbls1, lbls2

        del testEntries, trainEntries, n, totalEntries

        return imgs, lbls