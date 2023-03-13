import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

#Importante: Assume que os geradores recebidos já foram inicializados, compilados e treinados
class MIXED:
    def __init__(self, params, arr, ids, nameComplement = ""):
        self.name = '_'.join([arr[id].name for id in ids]) + '_' + self.__class__.__name__ + nameComplement

        self.currentFold = params.currentFold
        self.basePath = verifiedFolder('runtime/trainingStats/' + self.name)
        self.arr = arr
        self.ids = ids

    #compilando
    def compile(self):
        pass

    #treinamento
    def train(self, imgs, lbls):
        pass

    #Gera e salva imagens
    def saveGenerationExample(self, nEntries = 20):
        pass

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