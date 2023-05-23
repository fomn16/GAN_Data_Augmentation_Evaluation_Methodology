from Modules.Shared.helper import *
from Modules.Shared.config import *
from Modules.Datasets.MNIST import MNIST
from Modules.Datasets.CIFAR_10 import CIFAR_10
from Modules.Datasets.SOP import STANFORD_ONLINE_PRODUCTS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#carrega dataset
params = Params()
#local no qual os datasets serão salvos
params.dataDir = './tfDatasets'
#validação cruzada
params.kFold = 5
params.currentFold = 0
params.runtime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
params.saveModels = False

datasets : List[Dataset] = []
#datasets.append(MNIST(params))
datasets.append(CIFAR_10(params))
#datasets.append(STANFORD_ONLINE_PRODUCTS(params))

for fold in range(params.kFold):
    for dataset in datasets:
        params.currentFold = fold
        dataset.loadParams()

        augmentators : List[Augmentator] = []
        augmentators.extend(getAugmentators(Augmentators.CGAN, params))
        #augmentators.extend(getAugmentators(Augmentators.DIRECT, params))
        #augmentators.extend(getAugmentators(Augmentators.GAN, params))
        #augmentators.extend(getAugmentators(Augmentators.MIXED, params, [augmentators, {0,2}]))

        for augmentator in augmentators:
            if(augmentator != None):
                #treinando gan
                augmentator.compile()
                augmentator.train(dataset)

                #salva resultado final
                augmentator.saveGenerationExample()

                #cria testes
                benchmarks : List[Benchmark] = []
                benchmarks.extend(getBenchmarks(Benchmarks.CLASSIFIER, params))
                benchmarks.extend(getBenchmarks(Benchmarks.TSNE_INCEPTION, params))

                #percorre os testes
                for benchmark in benchmarks:
                    if(benchmark != None):
                        benchmark.train(augmentator, dataset)
                        benchmark.runTest(dataset)