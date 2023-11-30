import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

from Modules.Augmentation.Augmentator import Augmentator
from Modules.Shared.Params import Params
from Modules.Datasets.Dataset import Dataset

#Importante: Assume que os geradores recebidos já foram inicializados, compilados e treinados
class MIXED(Augmentator):
    def __init__(self, params:Params, extraParams = None, nameComplement = ""):
        arr = extraParams[0]
        ids = extraParams[1]

        if(len(extraParams) > 2 and extraParams[2] != None):
            self.percentages = extraParams[2]
        else:
            self.percentages = [1] * len(ids)

        self.name = self.__class__.__name__ + addToName(nameComplement) + '/' + '_'.join([arr[id].name for id in ids])

        self.currentFold = params.currentFold
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)
        self.arr = arr
        self.ids = ids

    def generate(self, srcImgs, srcLbls):
        imgs = []
        lbls = []
        
        for i, id in enumerate(self.ids):
            #filtrando a porcentagem requisitada do augmentator de origem
            (_srcImgs, _srcLbls) = shuffle(srcImgs, srcLbls)
            n = int(_srcLbls.shape[0] * self.percentages[i])
            _imgs, _lbls = self.arr[id].generate(_srcImgs[:n], _srcLbls[:n])

            if(len(imgs) == 0):
                imgs = _imgs
                lbls = _lbls
            else:
                imgs = np.concatenate((imgs, _imgs))
                lbls = np.concatenate((lbls, _lbls))

        (imgs,lbls) = shuffle(imgs,lbls)
        return imgs, lbls

    def verifyInitialization(self, dataset: Dataset):
        for id in self.ids:
            self.arr[id].verifyInitialization(dataset)