import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Shared.Params import Params

class DATASET_DIRECTLY(Augmentator):
    def __init__(self, params:Params, extraParams = None, nameComplement = ""):
        self.name = self.__class__.__name__ + "_" + params.datasetName + "_" + nameComplement

        self.currentFold = params.currentFold
        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)
        
        self.params = params

        self.dataposition = 0

        self.dataset = None

    #treinamento
    def train(self, dataset: Dataset):
        print('started ' + self.name + ' training')
        self.dataset = dataset

    #Gera e salva imagens
    def saveGenerationExample(self, nEntries = 20):
        start = 0
        if(nEntries != self.dataset.trainInstances):
            start = np.random.randint(0, self.dataset.trainInstances - nEntries)
        images, labels = self.dataset.getTrainData(start, start+nEntries)
        out = ((images * 127.5) + 127.5).astype('uint8')
        showOutputAsImg(out, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a.argmax()) for a in labels]) + '.png',nEntries, self.params.imgChannels == 3)

    def generate(self, srcImgs, srcLbls):
        nEntries = srcLbls.shape[0]
        if(self.dataposition + nEntries >= self.dataset.trainInstances):
            self.dataposition = 0
        img, data = self.dataset.getTrainData(self.dataposition, self.dataposition+nEntries)
        return img.copy(), data.copy()
    
    def verifyInitialization(self, dataset: Dataset):
        self.dataset = dataset