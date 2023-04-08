#Z:\felip\Documents\UNB\TCC\modulos
from Modules.Shared.helper import *
from Modules.Augmentation.GAN_MNIST import GAN_MNIST
from Modules.Augmentation.CGAN_MNIST import CGAN_MNIST
from Modules.Augmentation.DATASET_DIRECTLY import DATASET_DIRECTLY
from Modules.Augmentation.MIXED import MIXED
from Modules.Shared.Params import Params
from Modules.Benchmark.Classifier_MNIST import Classifier_MNIST

#carrega dataset
datasetName = 'mnist'
dataset, info = tfds.load(name = datasetName, with_info=True, as_supervised=True, data_dir='./tfDatasets')

params = Params()
params.nClasses = 10

params.imgChannels = 1
params.imgWidth = 28
params.imgHeight = 28

#validação cruzada
params.kFold = 5
params.currentFold = 4

#preparando dados
imgs1, lbls1 = loadIntoArray(dataset['train'], params.nClasses)
imgs2, lbls2 = loadIntoArray(dataset['test'], params.nClasses)
imgs = np.concatenate((imgs1, imgs2))
lbls = np.concatenate((lbls1, lbls2))

totalEntries = imgs.shape[0]

n = int(np.floor(totalEntries/params.kFold))

for f in range(params.kFold):
    params.currentFold = f
    trainImgs = imgs[:params.currentFold*n]
    trainLbls = lbls[:params.currentFold*n]
    testImgs = imgs[params.currentFold*n:(params.currentFold + 1)*n]
    testLbls = lbls[params.currentFold*n:(params.currentFold + 1)*n]
    trainImgs = np.concatenate((trainImgs, imgs[(params.currentFold + 1)*n:]))
    trainLbls = np.concatenate((trainLbls, lbls[(params.currentFold + 1)*n:]))

    generators = []
    generators.append(DATASET_DIRECTLY(params, testImgs, testLbls))
    generators.append(CGAN_MNIST(params))
    generators.append(GAN_MNIST(params))
    generators.append(MIXED(params, generators, {0,1}))

    #cria testes
    classifiers = []
    classifiers.append(Classifier_MNIST(params))

    for generator in generators:
        #treinando gan
        generator.compile()
        generator.train(trainImgs, trainLbls)

        #salva resultado final
        generator.saveGenerationExample()

        #gera dados para testes
        genImgs, genLbls = generator.generate(trainImgs.shape[0])

        #percorre os testes
        for classifier in classifiers:
            classifier.train(genImgs, genLbls, generator.name)
            classifier.runTest(testImgs, testLbls)


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