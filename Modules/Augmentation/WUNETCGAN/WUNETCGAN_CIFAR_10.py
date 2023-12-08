import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.WUNETCGAN.WUNETCGAN import WUNETCGAN

class WUNETCGAN_CIFAR_10(WUNETCGAN):
    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 8e-4
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
        self.similarityLossAmmount = 1/2 if "unbalanced" in self.params.datasetNameComplement else 1/5
        self.similarityLossDecaySteps = 1/10
        self.similarityLossDecayRate = 0.93

    def UNetCall(self, model):
        return self.UNet(model, 32, 1.5, 3, dropout=False)
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 32, dropout=False)
        model = self.TransposedBlock(model, 1, 16, dropout=False)
        model = self.TransposedBlock(model, 1, 8, dropout=False)
        return model
    
    def discDownscale(self, model):
        model = self.InceptionBlock(model, 3, 64, stride=2, dropout=False)
        model = self.InceptionBlock(model, 3, 64, stride=2, dropout=False)
        model = self.InceptionBlock(model, 3, 128, stride=2, dropout=False)
        return model