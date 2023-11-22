import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Benchmark.Benchmark import Benchmark

class Classifier_MNIST(Benchmark):
    leakyReluAlpha = 0.2
    FCOutputDim = 512
    initLr = 2e-4
    nEpochs = 10
    batchSize = 128

    def __init__(self, params: Params, nameComplement = ""):
        self.name = self.__class__.__name__ + "_" +  nameComplement

        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)
        self.currentFold = params.currentFold

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight

        self.params = params

        #faz um classificador com pesos iniciais a serem usados em todos os testes seguintes e o salva. Próximas chamadas irão apenas carregar este
        self.__create()

    def __create(self):
        classInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'classinput')
        classX = layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', strides=(2,2), activation='relu')(classInput)

        classX = layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', strides=(2,2), activation='relu')(classX)

        classX = layers.Flatten()(classX)

        classOutput = layers.Dense(self.nClasses, activation='sigmoid', name='genOutput_label')(classX)

        self.classifier = keras.Model(inputs = classInput, outputs = classOutput, name = 'classifier')

        keras.utils.plot_model(
            self.classifier, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '.png')
        )

        optClass = Adam(learning_rate = self.initLr, beta_1 = 0.5)
        self.classifier.compile(loss='categorical_crossentropy', optimizer=optClass)

        self.classifier.save(verifiedFolder(self.basePath + "/modelSaves/init/fold_" + str(self.currentFold)))

    def train(self, augmentator: Augmentator, dataset: Dataset):
        trainName = augmentator.name
        
        self.classifier = load_model(self.basePath + "/modelSaves/init/fold_" + str(self.currentFold))
        classLossHist = []
        print("Fold " + str(self.currentFold) + ": Starting Classifier training on " + trainName + "\n")
        infoFile = open(self.basePath + '/info.txt', 'a')
        infoFile.write("Fold " + str(self.currentFold) + ": Starting Classifier training on " + trainName)
        infoFile.close()

        nBatches = int(dataset.trainInstances/self.batchSize)
        for epoch in range(round(self.nEpochs)):
            imgs,lbls = augmentator.generate(dataset.trainImgs, dataset.trainLbls)
            lbls = np.array([[1 if i == lbl else 0 for i in range(self.nClasses)] for lbl in lbls], dtype='float32')
            for i in range(nBatches):
                imgBatch = imgs[i*self.batchSize:(i+1)*self.batchSize]
                labelBatch = lbls[i*self.batchSize:(i+1)*self.batchSize]

                classLoss = self.classifier.train_on_batch(imgBatch,labelBatch)
                classLossHist.append(classLoss)
                if i == nBatches-1:
                    IPython.display.clear_output(True)
                    
                    print("Epoch " + str(epoch) + "\nclassifier loss: " + str(classLoss))
                    infoFile = open(self.basePath + '/info.txt', 'a')
                    infoFile.write("Epoch " + str(epoch) + "\nclassifier loss: " + str(classLoss) + '\n')
                    infoFile.close()

                    plotLoss([[classLossHist, 'classifier loss']], self.basePath + '/trainPlot_f' + str(self.currentFold) + '_' + trainName + '.png')

            if(epoch % 5 == 0 or epoch == self.nEpochs - 1):
                self.classifier.save(verifiedFolder(self.basePath + '/modelSaves/' + trainName + '/fold_' + str(self.currentFold) + '/epoch_' + str(epoch)))
        self.classifier.save(verifiedFolder(self.basePath + '/modelSaves/' + trainName + '/fold_' + str(self.currentFold) + '/final'))

    def runTest(self, dataset: Dataset):
        imgs, lbls = dataset.getTestData(0, dataset.testInstances)
        lbls = np.array([[1 if i == lbl else 0 for i in range(self.nClasses)] for lbl in lbls], dtype='float32')
        classOutput = self.classifier.predict(imgs, verbose=0)
        classOutput = [[int(np.argmax(o) == i) for i in range(self.nClasses)] for o in classOutput]
        lbls = [[int(np.argmax(o) == i) for i in range(self.nClasses)] for o in lbls]
        
        aurocScore = ""
        try:
            aurocScore = str(roc_auc_score(lbls, classOutput))
        except:
            aurocScore = "error calculating:\n" + str(lbls) + "\n" + str(classOutput)
        
        report = classification_report(lbls, classOutput) + '\nauroc score: ' + aurocScore + '\n'

        print(report)
        infoFile = open(self.basePath + '/info.txt', 'a')
        infoFile.write(report)
        infoFile.close()