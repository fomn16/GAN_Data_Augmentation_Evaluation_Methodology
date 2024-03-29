import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params
from Modules.Shared.helper import *

from Modules.Benchmark.Classifier.Classifier_MNIST import Classifier_MNIST
from Modules.Benchmark.Classifier.Classifier_CIFAR import Classifier_CIFAR
from Modules.Benchmark.TSNE_INCEPTION import TSNE_INCEPTION
from Modules.Benchmark.Classifier.DiscriminatorAsClassifier import DiscriminatorAsClassifier

from Modules.Augmentation.GAN.GAN_MNIST import GAN_MNIST
from Modules.Augmentation.CGAN.CGAN_MNIST import CGAN_MNIST
from Modules.Augmentation.WCGAN.WCGAN_MNIST import WCGAN_MNIST
from Modules.Augmentation.WUNETCGAN.WUNETCGAN_MNIST import WUNETCGAN_MNIST

from Modules.Augmentation.GAN.GAN_CIFAR_10 import GAN_CIFAR_10
from Modules.Augmentation.CGAN.CGAN_CIFAR_10 import CGAN_CIFAR_10
from Modules.Augmentation.WCGAN.WCGAN_CIFAR_10 import WCGAN_CIFAR_10
from Modules.Augmentation.WUNETCGAN.WUNETCGAN_CIFAR_10 import WUNETCGAN_CIFAR_10

from Modules.Augmentation.WUNETCGAN.WUNETCGAN_PLANT import WUNETCGAN_PLANT

from Modules.Augmentation.GAN.GAN_EUROSAT import GAN_EUROSAT
from Modules.Augmentation.CGAN.CGAN_EUROSAT import CGAN_EUROSAT
from Modules.Augmentation.WCGAN.WCGAN_EUROSAT import WCGAN_EUROSAT
from Modules.Augmentation.WUNETCGAN.WUNETCGAN_EUROSAT import WUNETCGAN_EUROSAT

from Modules.Augmentation.WUNETCGAN.WUNETCGAN_QUICKDRAW import WUNETCGAN_QUICKDRAW

from Modules.Augmentation.DATASET_DIRECTLY import DATASET_DIRECTLY
from Modules.Augmentation.MIXED import MIXED

from Modules.Augmentation.Augmentator import Augmentator
from Modules.Benchmark.Benchmark import Benchmark

class Datasets:
    MNIST = "mnist"
    CIFAR_10 = "cifar10"
    PLANT = "plant_village"
    EUROSAT = "eurosat"
    QUICKDRAW = "quickdraw_bitmap"
    FLOWERS = "tf_flowers"
    IMAGENET = "imagenet_resized/32x32"
    TEST = "eurosat"

class Augmentators:
    GAN = "gan"
    CGAN = "cgan"
    WCGAN = "wcgan"
    WUNETCGAN = "wunetcgan"
    DIRECT = "dataset_directly"
    MIXED = "mixed"

class Benchmarks:
    CLASSIFIER = "classifier"
    TSNE_INCEPTION = "tsne_inception"
    DISC_AS_CLASSIFIER = "disc_as_classifier"

def getAugmentators(augmentator, params:Params, extraParams = None, nameComplement = "") -> List[Augmentator]:
    name = params.datasetNameComplement + addToName(nameComplement)
    if(augmentator == Augmentators.GAN):
        if(params.datasetName == Datasets.MNIST):
            return [GAN_MNIST(params, extraParams, name)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [GAN_CIFAR_10(params, extraParams, name)]
        if(params.datasetName == Datasets.PLANT):
            return [GAN_CIFAR_10(params, extraParams, "PLANT_"+name)]
        if(params.datasetName == Datasets.EUROSAT):
            return [GAN_EUROSAT(params, extraParams, name)]
    if(augmentator == Augmentators.CGAN):
        if(params.datasetName == Datasets.MNIST):
            return [CGAN_MNIST(params, extraParams, name)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [CGAN_CIFAR_10(params, extraParams, name)]
        if(params.datasetName == Datasets.PLANT):
            return [CGAN_CIFAR_10(params, extraParams, "PLANT_"+name)]
        if(params.datasetName == Datasets.EUROSAT):
            return [CGAN_EUROSAT(params, extraParams, name)]
    if(augmentator == Augmentators.WCGAN):
        if(params.datasetName == Datasets.MNIST):
            return [WCGAN_MNIST(params, extraParams, name)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [WCGAN_CIFAR_10(params, extraParams, name)]
        if(params.datasetName == Datasets.PLANT):
            return [WCGAN_CIFAR_10(params, extraParams, "PLANT_"+name)]
        if(params.datasetName == Datasets.EUROSAT):
            return [WCGAN_EUROSAT(params, extraParams, name)]
    if(augmentator == Augmentators.WUNETCGAN):
        if(params.datasetName == Datasets.MNIST):
            return [WUNETCGAN_MNIST(params, extraParams, name)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [WUNETCGAN_CIFAR_10(params, extraParams, name)]
        if(params.datasetName == Datasets.PLANT):
            return [WUNETCGAN_PLANT(params, extraParams, name)]
        if(params.datasetName == Datasets.EUROSAT):
            return [WUNETCGAN_EUROSAT(params, extraParams, name)]
        
        if(params.datasetName == Datasets.IMAGENET):
            return [WUNETCGAN_CIFAR_10(params, extraParams, "IMAGENET_"+name)]
        if(params.datasetName == Datasets.QUICKDRAW):
            return [WUNETCGAN_QUICKDRAW(params, extraParams, name)]
        if(params.datasetName == Datasets.TEST):
            return [WUNETCGAN_CIFAR_10(params, extraParams, name)]
    if(augmentator == Augmentators.DIRECT):
        return [DATASET_DIRECTLY(params, extraParams, name)]
    if(augmentator == Augmentators.MIXED):
        return [MIXED(params, extraParams, name)]
    return [None]

def getBenchmarks(benchmark, params: Params, nameComplement = "") -> List[Benchmark]:
    name = params.datasetNameComplement + addToName(nameComplement)
    if(benchmark == Benchmarks.CLASSIFIER):
        if(params.datasetName == Datasets.MNIST):
            return [Classifier_MNIST(params, name)]
        if(params.datasetName == Datasets.CIFAR_10):
            return [Classifier_CIFAR(params, name)]
        if(params.datasetName == Datasets.PLANT):
            return [Classifier_CIFAR(params,  "PLANT_"+name)]
        if(params.datasetName == Datasets.EUROSAT):
            return [Classifier_CIFAR(params,  "EUROSAT_"+name)]
    elif(benchmark == Benchmarks.TSNE_INCEPTION):
            return [TSNE_INCEPTION(params, name)]
    elif(benchmark == Benchmarks.DISC_AS_CLASSIFIER):
            return [DiscriminatorAsClassifier(params, name)]
    return [None]