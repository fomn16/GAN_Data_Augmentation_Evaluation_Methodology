import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.WUNETCGAN.WUNETCGAN import WUNETCGAN

class WUNETCGAN_QUICKDRAW(WUNETCGAN):
    def loadConstants(self):
        self.genWidth = 7
        self.genHeight = 7

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 2e-5
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

        self.wrongClassAmmt = 0.5
        self.similarityLossAmmount = 1/2
        self.similarityLossDecaySteps = 1/10
        self.similarityLossDecayRate = 0.93

    def UNetCall(self, model):
        return self.UNet(model, 32, 2, 2, dropout=False)
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 16)
        model = self.TransposedBlock(model, 1, 8)
        return model
    
    def discDownscale(self, model):
        model = self.ResidualBlock(model, 3, 32, stride=2)
        model = self.ResidualBlock(model, 3, 64, stride=2)
        model = self.ResidualBlock(model, 3, 128)
        return model
    
    def embeddingProcessing(self, model):
        ret = layers.Embedding(self.nClasses, self.genWidth*self.genHeight*5)(model)
        ret = layers.Dense(self.genWidth*self.genHeight*5)(ret)
        ret = layers.LeakyReLU(alpha=self.leakyReluAlpha)(ret)
        ret = layers.Dense(self.genWidth*self.genHeight*5)(ret)
        ret = layers.LeakyReLU(alpha=self.leakyReluAlpha)(ret)
        ret = layers.Reshape((self.genWidth, self.genHeight, 5))(ret)
        return ret