import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

#Importante: Assume que os geradores recebidos já foram inicializados, compilados e treinados
class MIXED(Augmentator):
    def __init__(self, params:Params, extraParams = None, nameComplement = ""):
        arr = extraParams[0]
        ids = extraParams[1]
        self.name = '_'.join([arr[id].name for id in ids]) + '_' + self.__class__.__name__ + nameComplement

        self.currentFold = params.currentFold
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)
        self.arr = arr
        self.ids = ids

    def generate(self, nEntries):
        imgs = []
        lbls = []
        
        for id in self.ids:
            _imgs, _lbls = self.arr[id].generate(nEntries)
            if(imgs == []):
                imgs = _imgs
                lbls = _lbls
            else:
                imgs = np.concatenate((imgs, _imgs))
                lbls = np.concatenate((lbls, _lbls))

        (imgs,lbls) = shuffle(imgs,lbls)
        return imgs, lbls