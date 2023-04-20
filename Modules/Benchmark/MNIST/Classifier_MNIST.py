import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

class Classifier_MNIST(Benchmark):
    #constantes
    leakyReluAlpha = 0.2
    FCOutputDim = 512
    initLr = 2e-4
    nEpochs = 8
    batchSize = 128

    classifier = None

    def __init__(self, params: Params, nameComplement = ""):
        self.name = self.__class__.__name__ + nameComplement

        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)
        self.currentFold = params.currentFold

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight

        self.params = params
        #infoFile = open(self.basePath + '/info.txt', 'w')
        #infoFile.close()

        #faz um classificador com pesos iniciais a serem usados em todos os testes seguintes e o salva. Próximas chamadas irão apenas carregar este
        self.__create()

    def __create(self):
        classInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'classinput')
        # primeira camada convolucional, recebe formato das imagens
        classX = layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', strides=(2,2))(classInput)
        classX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(classX)

        # segunda camada convolucional.
        classX = layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', strides=(2,2))(classX)
        classX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(classX)

        # camada densa
        classX = layers.Flatten()(classX)
        classX = layers.Dense(self.FCOutputDim)(classX)
        classX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(classX)

        # camada de output, chassificador one hot
        classOutput = layers.Dense(self.nClasses, activation='tanh', name='genOutput_label')(classX)

        self.classifier = keras.Model(inputs = classInput, outputs = classOutput, name = 'classifier')

        keras.utils.plot_model(
            self.classifier, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '.png')
        )

        optClass = Adam(learning_rate = self.initLr, beta_1 = 0.5)#, decay = self.initLr/self.nEpochs)
        self.classifier.compile(loss='binary_crossentropy', optimizer=optClass)
        self.classifier.save(verifiedFolder(self.basePath + "/modelSaves/init/fold_" + str(self.currentFold)))

    def create(self):
        self.classifier = load_model(self.basePath + "/modelSaves/init/fold_" + str(self.currentFold))

    def train(self, augmentator: Augmentator, dataset: Dataset, extraEpochs = 1):
        ''', aug = False):'''
        trainName = augmentator.name
        
        self.create()
        classLossHist = []
        print("Fold " + str(self.currentFold) + ": Starting Classifier training on " + trainName + "\n")
        infoFile = open(self.basePath + '/info.txt', 'a')
        infoFile.write("Fold " + str(self.currentFold) + ": Starting Classifier training on " + trainName)
        infoFile.close()

        '''seq = iaa.Sequential([
            #iaa.Crop(px=(0, 5)),
            #iaa.Fliplr(0.5),
            iaa.GaussianBlur(sigma=(0, 2.0))
        ])'''

        nBatches = int(dataset.trainInstances/self.batchSize)
        for epoch in range(round(self.nEpochs * extraEpochs)):
            imgs,lbls = augmentator.generate(self.batchSize*nBatches)
            for i in range(nBatches):
                imgBatch = imgs[i*nBatches:(i+1)*nBatches]
                labelBatch = lbls[i*nBatches:(i+1)*nBatches]
                #imgBatch, labelBatch = augmentator.generate(self.batchSize)
                '''if(aug):
                    imgBatch = seq(images=imgBatch)'''
                
                classLoss = self.classifier.train_on_batch(imgBatch,labelBatch)
                
                if i == nBatches-1:
                    classLossHist.append(classLoss)
                    IPython.display.clear_output(True)
                    
                    print("Epoch " + str(epoch) + "\nclassifier loss: " + str(classLoss))
                    infoFile = open(self.basePath + '/info.txt', 'a')
                    infoFile.write("Epoch " + str(epoch) + "\nclassifier loss: " + str(classLoss) + '\n')
                    infoFile.close()

                    plotLoss([[classLossHist, 'classifier loss']], self.basePath + '/trainPlot_f' + str(self.currentFold) + '_' + trainName + '.png')

            if(epoch % 5 == 0 or epoch == self.nEpochs * extraEpochs - 1):
                self.classifier.save(verifiedFolder(self.basePath + '/modelSaves/' + trainName + '/fold_' + str(self.currentFold) + '/epoch_' + str(epoch)))
        self.classifier.save(verifiedFolder(self.basePath + '/modelSaves/' + trainName + '/fold_' + str(self.currentFold) + '/final'))

    def runTest(self, dataset: Dataset):
        imgs, lbls = dataset.getAllTestData()
        classOutput = self.classifier.predict(imgs, verbose=0)
        classOutput = [[int(np.argmax(o) == i) for i in range(self.nClasses)] for o in classOutput]
        report = classification_report(lbls, classOutput) + '\nauroc score: ' + str(roc_auc_score(lbls, classOutput)) + '\n'

        print(report)
        infoFile = open(self.basePath + '/info.txt', 'a')
        infoFile.write(report)
        infoFile.close()