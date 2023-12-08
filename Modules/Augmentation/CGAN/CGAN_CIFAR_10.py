import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.CGAN.CGAN import CGAN

class CGAN_CIFAR_10(CGAN):
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

        self.ganEpochs = 50
        self.batchSize = 128
        self.extraDiscEpochs = 1
        self.generator = None
        self.discriminator = None
        self.gan = None

    def genUpscale(self, model):
        model = self.TransposedBlock(model, 2, 64)
        model = self.TransposedBlock(model, 2, 64)
        model = self.TransposedBlock(model, 1, 64)
        return model
    
    def discDownscale(self, model):
        model = self.ResidualBlock(model, 1, 64, stride=2)
        model = self.ResidualBlock(model, 1, 64, stride=2)
        model = self.ResidualBlock(model, 1, 64, stride=2)
        return model
