from Modules.Shared.helper import *
from Modules.Shared.config import *
from Modules.Datasets.MNIST import MNIST
from Modules.Datasets.CIFAR_10 import CIFAR_10
from Modules.Datasets.SOP import STANFORD_ONLINE_PRODUCTS
from Modules.Shared.Saving import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#carrega dataset
params = Params()

if(loadParam('active') is None):
    #local no qual os datasets serão salvos
    params.dataDir = './tfDatasets'
    #validação cruzada
    params.kFold = 5
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
#datasets.append(MNIST(params))
datasets.append(CIFAR_10(params))
#datasets.append(STANFORD_ONLINE_PRODUCTS(params))

for fold in range(params.currentFold, params.kFold):
    params.currentFold = fold
    saveParam('params_currentFold', params.currentFold)
    currentDatasetId = loadParam('current_dataset_id')
    if(currentDatasetId is None):
        currentDatasetId = 0
    for datasetId in range(currentDatasetId, len(datasets)):
        saveParam('current_dataset_id', datasetId)
        dataset = datasets[datasetId]

        dataset.loadParams()
        augmentators : List[Augmentator] = []
        augmentators.extend(getAugmentators(Augmentators.CGAN, params))
        augmentators.extend(getAugmentators(Augmentators.DIRECT, params))
        #augmentators.extend(getAugmentators(Augmentators.GAN, params))
        augmentators.extend(getAugmentators(Augmentators.MIXED, params, [augmentators, {0,1}]))

        currentAugmentatorId = loadParam('current_augmentator_id')
        if(currentAugmentatorId is None):
            currentAugmentatorId = 0
        for augmentatorId in range(currentAugmentatorId, len(augmentators)):
            saveParam('current_augmentator_id', augmentatorId)
            if(augmentators[augmentatorId] != None):
                #treinando gan
                augmentators[augmentatorId].compile()
                augmentators[augmentatorId].train(dataset)

                #salva resultado final
                augmentators[augmentatorId].saveGenerationExample()
                params.continuing = False

                #cria testes
                benchmarks : List[Benchmark] = []
                benchmarks.extend(getBenchmarks(Benchmarks.CLASSIFIER, params))
                benchmarks.extend(getBenchmarks(Benchmarks.TSNE_INCEPTION, params))

                #percorre os testes
                for benchmark in benchmarks:
                    if(benchmark != None):
                        benchmark.train(augmentators[augmentatorId], dataset)
                        benchmark.runTest(dataset)