import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.WCGAN.WCGAN import WCGAN

class WCGAN_CIFAR_10(WCGAN):
    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 2e-4
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.02
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
        model = self.TransposedBlock(model, 3, 64, dropout=False)
        model = self.TransposedBlock(model, 3, 64, kernelSize=5)
        model = self.TransposedBlock(model, 3, 64, dropout=False)
        return model
    
    def discDownscale(self, model):
        model = self.ResidualBlock(model, 4, 64, stride= 2, dropout=False)
        model = self.ResidualBlock(model, 4, 128, stride= 2)
        model = self.ResidualBlock(model, 3, 256, stride=2, dropout=False)
        return model
