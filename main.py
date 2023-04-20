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

datasets : List[Dataset] = []
#datasets.append(MNIST(params))
#datasets.append(CIFAR_10(params))
datasets.append(STANFORD_ONLINE_PRODUCTS(params))

for dataset in datasets:
    for fold in range(params.kFold):
        params.currentFold = fold
        dataset.loadParams()

        augmentators : List[Augmentator] = []
        #augmentators.append(getAugmentator(Augmentators.DIRECT, params))
        augmentators.append(getAugmentator(Augmentators.GAN, params))
        #augmentators.append(getAugmentator(Augmentators.CGAN, params))
        #augmentators.append(getAugmentator(Augmentators.MIXED, params, [augmentators, {0,2}]))

        #cria testes
        benchmarks : List[Benchmark] = []
        benchmarks.append(getBenchmark(Benchmarks.CLASSIFIER, params))
        benchmarks.append(getBenchmark(Benchmarks.TSNE_INCEPTION, params))

        for augmentator in augmentators:
            if(augmentator != None):
                #treinando gan
                augmentator.compile()
                augmentator.train(dataset)

                #salva resultado final
                augmentator.saveGenerationExample()

                #percorre os testes
                for benchmark in benchmarks:
                    if(benchmark != None):
                        benchmark.train(augmentator, dataset)
                        benchmark.runTest(dataset)

'''
gen.save(verifiedFolder('modelSaves/' + datasetName + '/gen/fold_' + str(currentFold)))
disc.save(verifiedFolder('modelSaves/' + datasetName + '/disc/fold_' + str(currentFold)))
gan.save(verifiedFolder('modelSaves/' + datasetName + '/gan/fold_' + str(currentFold)))
gen = load_model(verifiedFolder('modelSaves/' + datasetName + '/gen/fold_' + str(currentFold)))
disc = load_model(verifiedFolder('modelSaves/' + datasetName + '/disc/fold_' + str(currentFold)))
gan = load_model(verifiedFolder('modelSaves/' + datasetName + '/gan/fold_' + str(currentFold)))
'''

'''
cgen.save(verifiedFolder('modelSaves/' + datasetName + '/cgen/fold_' + str(currentFold)))
cdisc.save(verifiedFolder('modelSaves/' + datasetName + '/cdisc/fold_' + str(currentFold)))
cgan.save(verifiedFolder('modelSaves/' + datasetName + '/cgan/fold_' + str(currentFold)))

cgen = load_model(verifiedFolder('modelSaves/' + datasetName + '/cgen/fold_' + str(currentFold)))
cdisc = load_model(verifiedFolder('modelSaves/' + datasetName + '/cdisc/fold_' + str(currentFold)))
cgan = load_model(verifiedFolder('modelSaves/' + datasetName + '/cgan/fold_' + str(currentFold)))'''