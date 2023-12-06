import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.WCGAN.WCGAN import WCGAN

class WCGAN_MNIST(WCGAN):
    def loadConstants(self):
        self.genWidth = 7
        self.genHeight = 7

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 4e-4
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.05
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.clipValue = 0.01

        self.ganEpochs = 75
        self.batchSize = 128
        self.extraDiscEpochs = 5
        self.generator = None
        self.discriminator = None
        self.gan = None

    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 64, dropout=False, kernelSize=5)
        model = self.TransposedBlock(model, 1, 64, dropout=False)
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 2, 64, dropout=False, kernelSize=5)
        model = self.Block(model, 2, 128, dropout=False)
        return model