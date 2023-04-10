#Z:\felip\Documents\UNB\TCC\modulos
from Modules.Shared.helper import *
from Modules.Datasets.MNIST import MNIST
from Modules.Shared.Params import Params
from Modules.Benchmark.Classifier_MNIST import Classifier_MNIST
from Modules.Benchmark.TSNE_MNIST import TSNE_MNIST
from Modules.Augmentation.AugImplSelector import getAugmentator
from Modules.Shared.config import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#carrega dataset
params = Params()
#local no qual os datasets serão salvos
params.dataDir = './tfDatasets'
#validação cruzada
params.kFold = 5
params.currentFold = 0

datasets = []
datasets.append(MNIST(params))

for dataset in datasets:
    for fold in range(params.kFold):
        params.currentFold = fold
        dataset.loadParams()

        generators = []
        generators.append(getAugmentator(Augmentators.CGAN, params))
        generators.append(getAugmentator(Augmentators.DIRECT, params))
        generators.append(getAugmentator(Augmentators.GAN, params))
        generators.append(getAugmentator(Augmentators.MIXED, params, generators, {0,1}))

        #cria testes
        testers = []
        testers.append(Classifier_MNIST(params))
        testers.append(TSNE_MNIST(params))

        for generator in generators:
            #treinando gan
            generator.compile()
            generator.train(dataset)

            #salva resultado final
            generator.saveGenerationExample()

            #percorre os testes
            for tester in testers:
                tester.train(generator, dataset)
                tester.runTest(dataset.getAllTestData())


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