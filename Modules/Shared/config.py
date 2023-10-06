import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *

from Modules.Benchmark.Classifier.Classifier_MNIST import Classifier_MNIST
from Modules.Benchmark.Classifier.Classifier_CIFAR import Classifier_CIFAR
from Modules.Benchmark.Classifier.Classifier_SOP import Classifier_SOP
from Modules.Benchmark.TSNE_INCEPTION import TSNE_INCEPTION

from Modules.Augmentation.MNIST.GAN_MNIST import GAN_MNIST
from Modules.Augmentation.MNIST.CGAN_MNIST import CGAN_MNIST
from Modules.Augmentation.MNIST.AUGCGAN_MNIST import AUGCGAN_MNIST

from Modules.Augmentation.CIFAR_10.GAN_CIFAR_10 import GAN_CIFAR_10
from Modules.Augmentation.CIFAR_10.CGAN_CIFAR_10 import CGAN_CIFAR_10
from Modules.Augmentation.CIFAR_10.CGAN_CIFAR_10_Crossentropy import CGAN_CIFAR_10_Crossentropy
from Modules.Augmentation.CIFAR_10.AUGCGAN_CIFAR_10 import AUGCGAN_CIFAR_10

from Modules.Augmentation.SOP.GAN_SOP import GAN_SOP
from Modules.Augmentation.SOP.CGAN_SOP import CGAN_SOP

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

def getAugmentators(augmentator, params:Params, extraParams = None, nameComplement = "") -> List[Augmentator]:
    if(augmentator == Augmentators.GAN):
        if(params.datasetName == Datasets.MNIST):
            return [GAN_MNIST(params, extraParams, nameComplement)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [GAN_CIFAR_10(params, extraParams, nameComplement)]
        if(params.datasetName == Datasets.STANFORD_ONLINE):
            return [GAN_SOP(params, extraParams, nameComplement)]
    elif(augmentator == Augmentators.CGAN):
        if(params.datasetName == Datasets.MNIST):
            return [AUGCGAN_MNIST(params, extraParams, nameComplement)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [CGAN_CIFAR_10(params, extraParams, nameComplement)]#AUGCGAN_CIFAR_10(params, extraParams, nameComplement)]#CGAN_RESNET_CIFAR_10(params, extraParams, nameComplement)]#, CGAN_CIFAR_10_Crossentropy(params, extraParams, nameComplement)]
        if(params.datasetName == Datasets.STANFORD_ONLINE):
            return [CGAN_SOP(params, extraParams, nameComplement)]
    elif(augmentator == Augmentators.DIRECT):
        return [DATASET_DIRECTLY(params, extraParams, nameComplement)]
    elif(augmentator == Augmentators.MIXED):
        return [MIXED(params, extraParams, nameComplement)]
    return [None]

def getBenchmarks(benchmark, params: Params, nameComplement = "") -> List[Benchmark]:
    if(benchmark == Benchmarks.CLASSIFIER):
        if(params.datasetName == Datasets.MNIST):
            return [Classifier_MNIST(params)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [Classifier_CIFAR(params)]
        if(params.datasetName == Datasets.STANFORD_ONLINE):
            return [Classifier_SOP(params)]
    elif(benchmark == Benchmarks.TSNE_INCEPTION):
            return [TSNE_INCEPTION(params)]
    return [None]