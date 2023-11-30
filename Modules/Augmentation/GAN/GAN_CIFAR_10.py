import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.GAN.GAN import GAN

class GAN_CIFAR_10(GAN):
    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4
        self.genDepth = 128

        self.noiseDim = 100
        self.genFCOutputDim = 1024
        self.discFCOutputDim = 2048

        self.initLr = 2e-4
        self.leakyReluAlpha = 0.2
        self.l2RegParam = 0.01
        self.dropoutParam = 0.05
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.ganEpochs = 25
        self.batchSize = 64

        self.generator = None
        self.discriminator = None
        self.gan = None
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 128, kernelRegularizer=regularizers.l2(self.l2RegParam))
        model = self.TransposedBlock(model, 1, 128, kernelRegularizer=regularizers.l2(self.l2RegParam))
        model = self.TransposedBlock(model, 1, 128, kernelRegularizer=regularizers.l2(self.l2RegParam))
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 1, 64, kernelRegularizer=regularizers.l2(self.l2RegParam))
        model = self.Block(model, 1, 128, kernelRegularizer=regularizers.l2(self.l2RegParam))
        model = self.Block(model, 1, 256, kernelRegularizer=regularizers.l2(self.l2RegParam))
        model = self.Block(model, 1, 256, kernelRegularizer=regularizers.l2(self.l2RegParam))
        return model