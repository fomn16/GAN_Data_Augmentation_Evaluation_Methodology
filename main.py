#Z:\felip\Documents\UNB\TCC\modulos
from Modules.Shared.helper import *
from Modules.Datasets.MNIST import MNIST
from Modules.Augmentation.GAN_MNIST import GAN_MNIST
from Modules.Augmentation.CGAN_MNIST import CGAN_MNIST
from Modules.Augmentation.DATASET_DIRECTLY import DATASET_DIRECTLY
from Modules.Augmentation.MIXED import MIXED
from Modules.Shared.Params import Params
from Modules.Benchmark.Classifier_MNIST import Classifier_MNIST
from Modules.Benchmark.TSNE_MNIST import TSNE_MNIST

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
        generators.append(DATASET_DIRECTLY(params))
        generators.append(CGAN_MNIST(params))
        generators.append(GAN_MNIST(params))
        generators.append(MIXED(params, generators, {0,1}))

        #cria testes
        testers = []
        testers.append(TSNE_MNIST(dataset, params))
        testers.append(Classifier_MNIST(params))

        for generator in generators:
            #treinando gan
            generator.compile()
            generator.train(dataset.getTrainData())

            #salva resultado final
            generator.saveGenerationExample()

            #percorre os testes
            for tester in testers:
                tester.train(generator, dataset.getNTrain())
                tester.runTest(dataset.getTestData())


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