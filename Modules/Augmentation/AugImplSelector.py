import sys
sys.path.insert(1, '../../')
from Modules.Shared.config import *
from Modules.Augmentation.MNIST.CGAN_MNIST import CGAN_MNIST
from Modules.Augmentation.MNIST.GAN_MNIST import GAN_MNIST
from Modules.Augmentation.DATASET_DIRECTLY import DATASET_DIRECTLY
from Modules.Augmentation.MIXED import MIXED
from Modules.Shared.Params import Params

def getAugmentator(augmentator, params: Params, generators = None, ids = None):
    if(augmentator == Augmentators.GAN):
        if(params.datasetName == Datasets.MNIST):
            return GAN_MNIST(params)
    elif(augmentator == Augmentators.CGAN):
        if(params.datasetName == Datasets.MNIST):
            return CGAN_MNIST(params)
    elif(augmentator == Augmentators.DIRECT):
        return DATASET_DIRECTLY(params)
    elif(augmentator == Augmentators.MIXED):
        return MIXED(params, generators, ids)
    return None