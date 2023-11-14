import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *
from Modules.Augmentation.WUNETCGAN.WUNETCGAN import WUNETCGAN

class WUNETCGAN_CIFAR_10(WUNETCGAN):
    def __init__(self, params: Params, extraParams=None, nameComplement=""):
        super().__init__(params, extraParams)
        self.name = self.__class__.__name__+'_'+params.datasetName+'_'+params.datasetNameComplement+'_'+nameComplement

    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 2e-5
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.02
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.clipValue = 0.01

        self.ganEpochs = 100
        self.batchSize = 64
        self.extraDiscEpochs = 2
        self.generator = None
        self.discriminator = None
        self.gan = None 

        self.uNetChannels = 32
        self.uNetRatio = 1.5
        self.uNetBlocks = 3
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 2, 32)
        model = self.TransposedBlock(model, 2, 32)
        model = self.TransposedBlock(model, 2, 6)
        return model
    
    def discDownscale(self, model):
        model = self.InceptionBlock(model, 2, 32, stride=2)
        model = self.InceptionBlock(model, 2, 32, stride=2)
        model = self.InceptionBlock(model, 2, 32, stride=2)
        return model