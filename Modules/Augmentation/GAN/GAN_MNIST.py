import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.GAN.GAN import GAN

class GAN_MNIST(GAN):
    def loadConstants(self):
        self.genWidth = 7
        self.genHeight = 7
        self.genDepth = 64

        self.noiseDim = 100
        self.genFCOutputDim = 512
        self.discFCOutputDim = 512

        self.initLr = 2e-4
        self.leakyReluAlpha = 0.2
        self.l2RegParam = 0.01
        self.dropoutParam = 0.05
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.ganEpochs = 25
        self.batchSize = 128

        self.generator = None
        self.discriminator = None
        self.gan = None
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 64)
        model = self.TransposedBlock(model, 1, 64)
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 1, 64)
        model = self.Block(model, 1, 64)
        return model