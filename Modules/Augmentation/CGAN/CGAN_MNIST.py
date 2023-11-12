import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.CGAN.CGAN import CGAN

class CGAN_MNIST(CGAN):
    def __init__(self, params: Params, extraParams=None, nameComplement=""):
        super().__init__(params, extraParams, 'MNIST_' + nameComplement)

    def loadConstants(self):
        self.genWidth = 7
        self.genHeight = 7

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 2e-4
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.05
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.ganEpochs = 25
        self.batchSize = 128
        self.extraDiscEpochs = 1
        
        self.generator = None
        self.discriminator = None
        self.gan = None

    def genUpscale(self, model):
        model = self.TransposedBlock(model, 2, 64)
        model = self.TransposedBlock(model, 2, 64)
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 1, 64)
        model = self.Block(model, 1, 64)
        return model