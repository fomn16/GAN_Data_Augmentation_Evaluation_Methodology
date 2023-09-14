from Modules.Shared.helper import *
from Modules.Shared.config import *
from Modules.Datasets.MNIST import MNIST
from Modules.Datasets.CIFAR_10 import CIFAR_10
from Modules.Datasets.SOP import STANFORD_ONLINE_PRODUCTS
from Modules.Shared.Saving import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#cria objeto de parametros
params = Params()

#vendo se havia uma execução em andamento, e carregando seu progresso
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
datasets.append(CIFAR_10(params))
#datasets.append(MNIST(params))
#datasets.append(STANFORD_ONLINE_PRODUCTS(params))

for fold in range(params.currentFold, params.kFold):
    params.currentFold = fold
    saveParam('params_currentFold', params.currentFold)

    loadedDatasetId = loadParam('current_dataset_id', 0)
    for dataset in datasets[loadedDatasetId:]:
        saveParam('current_dataset_id', loadedDatasetId)
        loadedDatasetId+=1

        dataset.loadParams()

        augmentators : List[Augmentator] = []
        #augmentators.extend(getAugmentators(Augmentators.GAN, params))
        augmentators.extend(getAugmentators(Augmentators.CGAN, params))
        augmentators.extend(getAugmentators(Augmentators.DIRECT, params))
        augmentators.extend(getAugmentators(Augmentators.MIXED, params, [augmentators, {0,1}]))

        loadedAugmentatorId = loadParam('current_augmentator_id', 0)
        for augmentator in augmentators[loadedAugmentatorId:]:
            saveParam('current_augmentator_id', loadedAugmentatorId)
            loadedAugmentatorId+=1
            if(augmentator != None):

                #treinando
                augmentator.compile()
                augmentator.train(dataset)

                #salva resultado final
                augmentator.saveGenerationExample()
                params.continuing = False

                #cria testes
                benchmarks : List[Benchmark] = []
                benchmarks.extend(getBenchmarks(Benchmarks.TSNE_INCEPTION, params))
                benchmarks.extend(getBenchmarks(Benchmarks.CLASSIFIER, params))

                #percorre os testes
                for benchmark in benchmarks:
                    if(benchmark != None):
                        benchmark.train(augmentator, dataset)
                        benchmark.runTest(dataset)
        saveParam('current_augmentator_id', 0)
    saveParam('current_dataset_id', 0)