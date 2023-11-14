import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.CGAN.CGAN import CGAN

class CGAN_CIFAR_10(CGAN):
    def __init__(self, params: Params, extraParams=None, nameComplement=""):
        super().__init__(params, extraParams)
        self.name = self.__class__.__name__+'_'+params.datasetName+'_'+params.datasetNameComplement+'_'+nameComplement

    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 2.5e-5
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.05
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4
        self.l2RegParam = 0.01

        self.ganEpochs = 25
        self.batchSize = 32
        self.extraDiscEpochs = 1
        self.generator = None
        self.discriminator = None
        self.gan = None

    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 64, 3)
        model = self.TransposedBlock(model, 1, 64, 3)
        model = self.TransposedBlock(model, 1, 64, 3)
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 1, 64, 3)
        model = self.Block(model, 1, 64, 3)
        model = self.Block(model, 1, 64, 3)
        return model
