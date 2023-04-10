import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

class DATASET_DIRECTLY:
    def __init__(self, params, nameComplement = ""):
        self.name = self.__class__.__name__ + nameComplement

        self.currentFold = params.currentFold
        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime/trainingStats/' + self.name)

    #compilando discriminador e gan
    def compile(self):
        pass

    #treinamento
    def train(self, dataset):
        self.dataset = dataset

    #Gera e salva imagens
    def saveGenerationExample(self, nEntries = 20):
        start = np.random.randint(0, self.dataset.trainInstances - nEntries)
        images, labels = self.dataset.getTrainData(start, start+nEntries)
        out = ((images * 127.5) + 127.5).astype('uint8')
        showOutputAsImg(out, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a.argmax()) for a in labels]) + '.png',nEntries)

    def generate(self, nEntries):
        start = np.random.randint(0, self.dataset.trainInstances - nEntries)
        return self.dataset.getTrainData(start, start+nEntries)