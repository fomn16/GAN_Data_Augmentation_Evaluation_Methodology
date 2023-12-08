import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.CGAN.CGAN import CGAN

class CGAN_MNIST(CGAN):
    def loadConstants(self):
        self.genWidth = 7
        self.genHeight = 7

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 8e-5
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.2
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.ganEpochs = 75
        self.batchSize = 128
        self.extraDiscEpochs = 2
        
        self.generator = None
        self.discriminator = None
        self.gan = None

    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 64, kernelSize=5, dropout=False, batchNorm=True)
        model = self.TransposedBlock(model, 2, 64, kernelSize=5 ,dropout=False, batchNorm=True)
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 2, 64, kernelSize=5, batchNorm=True, dropout=False)
        model = self.Block(model, 1, 64, kernelSize=5, batchNorm=True, dropout=True)
        return model
    
    def discOutputProcessing(self, model):
        X = Conv2D(1, self.genWidth, kernel_initializer='glorot_uniform', activation='tanh')(model)
        X = Flatten(name = 'discoutput_realvsfake')(X)
        return X