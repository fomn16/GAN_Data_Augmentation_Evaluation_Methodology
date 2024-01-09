import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from Modules.Shared.helper import *
from Modules.Shared.config import *

from Modules.Datasets.Implementations.MNIST import MNIST
from Modules.Datasets.Implementations.MNIST_UNBALANCED import MNIST_UNBALANCED
from Modules.Datasets.Implementations.CIFAR_10 import CIFAR_10
from Modules.Datasets.Implementations.CIFAR_10_UNBALANCED import CIFAR_10_UNBALANCED
from Modules.Datasets.Implementations.PLANT import PLANT
from Modules.Datasets.Implementations.PLANT_UNBALANCED import PLANT_UNBALANCED
from Modules.Datasets.Implementations.EUROSAT import EUROSAT
from Modules.Datasets.Implementations.EUROSAT_UNBALANCED import EUROSAT_UNBALANCED

from Modules.Datasets.Implementations.TEST import TEST
from Modules.Datasets.Implementations.QUICKDRAW import QUICKDRAW
from Modules.Datasets.Implementations.FLOWERS import FLOWERS
from Modules.Datasets.Implementations.IMAGENET import IMAGENET

from Modules.Shared.Saving import *

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Benchmark.Benchmark import Benchmark

#cria objeto de parametros
params = Params()

#vendo se havia uma execução em andamento, e carregando seu progresso
if(loadParam('active') is None):
    #local no qual os datasets serão salvos
    params.dataDir = './tfDatasets'
    #validação cruzada
    params.kFold = 4
    params.currentFold = 0
    params.runtime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    params.saveModels = True
    params.continuing = False
    saveParam('params_dataDir', params.dataDir)
    saveParam('params_kFold', params.kFold)
    saveParam('params_currentFold', params.currentFold)
    saveParam('params_runtime', params.runtime)
    saveParam('params_saveModels', params.saveModels)
    saveParam('active', True)
    saveParam('close', False)
else:
    params.dataDir = loadParam('params_dataDir')
    params.kFold = loadParam('params_kFold')
    params.currentFold = loadParam('params_currentFold')
    params.runtime = loadParam('params_runtime')
    params.saveModels = loadParam('params_runtime')
    params.continuing = True

datasets : List[Dataset] = []

datasets.append(MNIST(params))
datasets.append(MNIST_UNBALANCED(params))
datasets.append(CIFAR_10(params))
datasets.append(CIFAR_10_UNBALANCED(params))
datasets.append(PLANT(params))
datasets.append(PLANT_UNBALANCED(params))
datasets.append(EUROSAT(params))
datasets.append(EUROSAT_UNBALANCED(params))

for fold in range(params.currentFold, params.kFold):
    params.currentFold = fold
    saveParam('params_currentFold', params.currentFold)
    loadedDatasetId = loadParam('current_dataset_id', 0)

    for dataset in datasets[loadedDatasetId:]:
        saveParam('current_dataset_id', loadedDatasetId)
        loadedDatasetId+=1
        dataset.load()

        def addWithMixTests(augList, name, params):
            augList.extend(getAugmentators(name, params))
            id = len(augList) - 1
            for i in range (10,100,10):
                n=i/100
                augList.extend(getAugmentators(Augmentators.MIXED, params, [augmentators, [0,id], [n,1-n]], str(i)+'_'+str(100-i)))
                augList.extend(getAugmentators(Augmentators.MIXED, params, [augmentators, [0,id], [1,n]], '100_'+str(i)))
                augList.extend(getAugmentators(Augmentators.MIXED, params, [augmentators, [0,id], [n,1]], str(i)+'_100'))
            augList.extend(getAugmentators(Augmentators.MIXED, params, [augmentators, [0,id], [1,1]], '100_100'))

        augmentators : List[Augmentator] = []
        augmentators.extend(getAugmentators(Augmentators.DIRECT, params))
        addWithMixTests(augmentators, Augmentators.GAN, params)
        addWithMixTests(augmentators, Augmentators.CGAN, params)
        addWithMixTests(augmentators, Augmentators.WCGAN, params)
        addWithMixTests(augmentators, Augmentators.WUNETCGAN, params)

        loadedAugmentatorId = loadParam('current_augmentator_id', 0)
        for augmentator in augmentators[loadedAugmentatorId:]:
            saveParam('current_augmentator_id', loadedAugmentatorId)
            loadedAugmentatorId+=1
            if(augmentator != None):
                #treinamento
                augmentator.compile()
                augmentator.train(dataset)

                #salva resultado final
                augmentator.saveGenerationExample()
                params.continuing = False
                #cria testes
                benchmarks : List[Benchmark] = []
                benchmarks.extend(getBenchmarks(Benchmarks.DISC_AS_CLASSIFIER, params))
                benchmarks.extend(getBenchmarks(Benchmarks.CLASSIFIER, params))
                if("MIXED" not in augmentator.name):
                    benchmarks.extend(getBenchmarks(Benchmarks.TSNE_INCEPTION, params))

                #percorre os testes
                for benchmark in benchmarks:
                    if(benchmark != None):
                        augmentator.verifyInitialization(dataset)
                        benchmark.train(augmentator, dataset)
                        benchmark.runTest(dataset)
        saveParam('current_augmentator_id', 0)
        dataset.unload()
    sys.exit()
    saveParam('current_dataset_id', 0)