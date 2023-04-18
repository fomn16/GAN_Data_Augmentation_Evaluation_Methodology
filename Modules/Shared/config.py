import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *

from Modules.Benchmark.MNIST.Classifier_MNIST import Classifier_MNIST
from Modules.Benchmark.TSNE_INCEPTION import TSNE_INCEPTION

from Modules.Augmentation.MNIST.CGAN_MNIST import CGAN_MNIST
from Modules.Augmentation.CIFAR_10.CGAN_CIFAR_10 import CGAN_CIFAR_10

from Modules.Augmentation.MNIST.GAN_MNIST import GAN_MNIST
from Modules.Augmentation.CIFAR_10.GAN_CIFAR_10 import GAN_CIFAR_10

from Modules.Augmentation.SOP.GAN_SOP import GAN_SOP

from Modules.Augmentation.DATASET_DIRECTLY import DATASET_DIRECTLY
from Modules.Augmentation.MIXED import MIXED

class Datasets:
    MNIST = "mnist"
    CIFAR_10 = "cifar10"
    STANFORD_ONLINE = "stanford_online_products"

class Augmentators:
    GAN = "gan"
    CGAN = "cgan"
    DIRECT = "dataset_directly"
    MIXED = "mixed"

class Benchmarks:
    CLASSIFIER = "classifier"
    TSNE_INCEPTION = "tsne_inception"

def getAugmentator(augmentator, params:Params, extraParams = None, nameComplement = "") -> Augmentator:
    if(augmentator == Augmentators.GAN):
        if(params.datasetName == Datasets.MNIST):
            return GAN_MNIST(params, extraParams, nameComplement)
        if(params.datasetName == Datasets.CIFAR_10):
            return GAN_CIFAR_10(params, extraParams, nameComplement)
        if(params.datasetName == Datasets.STANFORD_ONLINE):
            return GAN_SOP(params, extraParams, nameComplement)
    elif(augmentator == Augmentators.CGAN):
        if(params.datasetName == Datasets.MNIST):
            return CGAN_MNIST(params, extraParams, nameComplement)
        if(params.datasetName == Datasets.CIFAR_10):
            return CGAN_CIFAR_10(params, extraParams, nameComplement)
    elif(augmentator == Augmentators.DIRECT):
        return DATASET_DIRECTLY(params, extraParams, nameComplement)
    elif(augmentator == Augmentators.MIXED):
        return MIXED(params, extraParams, nameComplement)
    return None

def getBenchmark(benchmark, params: Params, nameComplement = "") -> Benchmark:
    if(benchmark == Benchmarks.CLASSIFIER):
        if(params.datasetName == Datasets.MNIST):
            return Classifier_MNIST(params)
        if(params.datasetName == Datasets.CIFAR_10):
            return Classifier_MNIST(params, "_CIFAR")
        if(params.datasetName == Datasets.STANFORD_ONLINE):
            return Classifier_MNIST(params, "_SOP")
    elif(benchmark == Benchmarks.TSNE_INCEPTION):
            return TSNE_INCEPTION(params)
    return None