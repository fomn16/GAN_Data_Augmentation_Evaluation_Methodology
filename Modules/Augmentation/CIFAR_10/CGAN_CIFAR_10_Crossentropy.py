import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Shared.Params import Params

class CGAN_CIFAR_10_Crossentropy(Augmentator):
    #Constantes:
    genWidth = 4
    genHeight = 4
    genDepth = 128

    noiseDim = 100
    genFCOutputDim = 1024
    discFCOutputDim = 2048

    initLr = 2e-4
    leakyReluAlpha = 0.2
    l2RegParamGen = 0.01
    l2RegParamDisc = 0.1
    dropoutParam = 0.01
    batchNormMomentum = 0.4

    ganEpochs = 100
    batchSize = 64

    generator = None
    discriminator = None
    gan = None

    def __init__(self, params: Params, extraParams = None, nameComplement = ""):
        self.name = self.__class__.__name__ + "_" +  nameComplement

        self.currentFold = params.currentFold
        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight

        self.params = params
    
    def AddBlock(self, inModel, nLayers: int, outDepth: int, kernelSize:int):
        for i in range(nLayers):
            if i == 0:
                model = layers.BatchNormalization(axis=-1, momentum=self.batchNormMomentum)(inModel)
                model = layers.Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_regularizer=regularizers.l2(self.l2RegParamDisc))(model)
            else:
                model = layers.Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_regularizer=regularizers.l2(self.l2RegParamDisc))(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
            model = layers.Dropout(self.dropoutParam)(model)
        model = layers.MaxPool2D(pool_size=(2,2), padding='valid', strides=(2,2))(model)
        return model
    
    def AddBlockTranspose(self, inModel, nLayers: int, outDepth: int, kernelSize:int):
        for i in range(nLayers):
            if i == 0:
                model = layers.BatchNormalization(axis=-1, momentum=self.batchNormMomentum)(inModel)
                model = layers.Conv2DTranspose(filters=outDepth, kernel_size=kernelSize, padding='same', strides=(2,2), kernel_regularizer=regularizers.l2(self.l2RegParamGen))(model)
            else:
                model = layers.Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_regularizer=regularizers.l2(self.l2RegParamGen))(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
            model = layers.Dropout(self.dropoutParam)(model)
        return model
    
    #Cria model geradora com keras functional API
    def createGenModel(self):
        cgenNoiseInput = keras.Input(shape=(self.noiseDim,), name = 'genInput_randomDistribution')

        # cria camada que converte saída da primeira camada para o número de nós necessário na entrada
        # das camadas convolucionais
        cgenX = layers.Dense(units=self.genWidth*self.genHeight*self.genDepth)(cgenNoiseInput)
        cgenX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(cgenX)
        cgenX = layers.Dropout(self.dropoutParam)(cgenX)
        #cgenX = layers.BatchNormalization()(cgenX)

        # Faz reshape para dimensões espaciais desejadas
        cgenX = layers.Reshape((self.genWidth, self.genHeight, self.genDepth))(cgenX)
    
        cgenLabelInput = keras.Input(shape=(1,), name = 'genInput_label')
        embeddedLabels= layers.Embedding(self.nClasses, self.genWidth*self.genHeight)(cgenLabelInput)
        reshapedLabels = layers.Reshape((self.genWidth, self.genHeight, 1))(embeddedLabels)
        cgenX = layers.concatenate([cgenX, reshapedLabels])

        model = self.AddBlockTranspose(cgenX, 1, 512, 3)

        model = self.AddBlockTranspose(model, 1, 256, 3)

        model = self.AddBlockTranspose(model, 2, 128, 3)

        cgenOutput = layers.Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='tanh',  name = 'genOutput_img', kernel_regularizer=regularizers.l2(self.l2RegParamGen))(model)
        
        self.generator = keras.Model(inputs = [cgenNoiseInput, cgenLabelInput], outputs = cgenOutput, name = 'cgenerator')

        keras.utils.plot_model(
            self.generator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/generator.png')
        )

    #Cria model discriminadora com functional API
    def createDiscModel(self):
        discInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'discinput_img')

        labelInput = keras.Input(shape=(1,), name = 'discinput_label')
        embeddedLabels = layers.Embedding(self.nClasses, self.imgWidth*self.imgHeight)(labelInput)
        reshapedLabels = layers.Reshape((self.imgWidth, self.imgHeight, 1))(embeddedLabels)
        discX = layers.concatenate([discInput, reshapedLabels])

        discX = self.AddBlock(discX, 2, 64, 3)
        discX = self.AddBlock(discX, 1, 128, 3)
        discX = self.AddBlock(discX, 1, 256, 3)
        discX = self.AddBlock(discX, 2, 512, 3)
        #discX = layers.BatchNormalization(axis=-1)(discX)

        # camada densa
        discX = layers.Flatten()(discX)

        # nó de output, mapear em 0 ou 1
        discOutput = layers.Dense(1, activation='sigmoid', name = 'discoutput_realvsfake')(discX)

        self.discriminator = keras.Model(inputs = [discInput, labelInput], outputs = discOutput, name = 'discriminator')

        keras.utils.plot_model(
            self.discriminator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/discriminator.png')
        )

    #compilando discriminador e gan
    def compile(self):
        optDiscr = Adam(learning_rate = self.initLr, beta_1 = 0.5)
        self.createDiscModel()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optDiscr, metrics=['accuracy'])

        self.createGenModel()
        self.discriminator.trainable = False
        cganNoiseInput = Input(shape=(self.noiseDim,))
        cganLabelInput = Input(shape=(1,))
        cganOutput =  self.discriminator([self.generator([cganNoiseInput, cganLabelInput]), cganLabelInput])
        self.gan = Model((cganNoiseInput, cganLabelInput), cganOutput)

        optcGan = Adam(learning_rate = self.initLr, beta_1=0.5)
        self.gan.compile(loss='binary_crossentropy', optimizer=optcGan)
        
        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )
        
    #treinamento GAN
    def train(self, dataset: Dataset):
        discLossHist = []
        genLossHist = []

        #noise e labels de benchmark
        benchNoise = np.random.uniform(-1,1, size=(256,self.noiseDim))
        benchLabels = np.random.randint(0,self.nClasses, size = (256))
        for i in range(20):
            benchLabels[i] = int(i/2)

        for epoch in range(self.ganEpochs):
            nBatches = int(dataset.trainInstances/self.batchSize)
            for i in range(nBatches):
                imgBatch, labelBatch = dataset.getTrainData((i)*self.batchSize, (i+1)*self.batchSize)
                
                genInput = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                labelInput = np.random.randint(0,self.nClasses, size = (self.batchSize))
                
                genImgOutput = self.generator.predict([genInput, labelInput], verbose=0)

                XImg = np.concatenate((imgBatch, genImgOutput))
                XLabel = np.concatenate((labelBatch, labelInput))
                y = ([0] * self.batchSize) + ([1] * self.batchSize)
                y = np.reshape(y, (-1,))
                (XImg, XLabel, y) = shuffle(XImg, XLabel, y)
                
                discLoss = self.discriminator.train_on_batch([XImg,XLabel], y)

                '''for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]
                    layer.set_weights(weights)'''
                
                genTrainNoise = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                genTrainClasses = np.random.randint(0,self.nClasses, size = (self.batchSize))
                #genTrainClasses = np.array([[1 if i == c else -1 for i in range(self.nClasses)] for c in genTrainClasses], dtype='float32')

                gentrainLbls = [0]*self.batchSize 
                gentrainLbls = np.reshape(gentrainLbls, (-1,))
                ganLoss = self.gan.train_on_batch([genTrainNoise, genTrainClasses],gentrainLbls)

                if i == nBatches-1:
                    discLossHist.append(discLoss)
                    genLossHist.append(ganLoss)

                    print("Epoch " + str(epoch) + "\nCGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss))
                    infoFile = open(self.basePath + '/info.txt', 'a')
                    infoFile.write("Epoch " + str(epoch) + "\nCGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss)+ '\n')
                    infoFile.close()

                    images = self.generator.predict([benchNoise, benchLabels])
                    out = ((images * 127.5) + 127.5).astype('uint8')

                    showOutputAsImg(out, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a) for a in benchLabels[:20]]) + '.png', colored=True)
                    plotLoss([[genLossHist, 'generator loss'],[discLossHist, 'discriminator loss']], self.basePath + '/trainPlot.png')

            if((self.params.saveModels and epoch%5 == 0) or epoch == self.ganEpochs-1):
                epochPath = self.basePath + '/modelSaves/fold_' + str(self.currentFold) + '/epoch_' + str(epoch)
                self.generator.save(verifiedFolder(epochPath + '/gen'))
                self.discriminator.save(verifiedFolder(epochPath + '/disc'))
                self.gan.save(verifiedFolder(epochPath + '/gan'))

    #Gera e salva imagens
    def saveGenerationExample(self, nEntries = 20):
        noise = np.random.uniform(-1,1, size=(self.nClasses,self.noiseDim))
        labels = np.array(range(self.nClasses))
        images = self.generator.predict([noise, labels])
        out = ((images * 127.5) + 127.5).astype('uint8')
        showOutputAsImg(out, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a) for a in labels]) + '.png',self.nClasses, colored=True)

    def generate(self, nEntries):
        print(self.name + ": started data generation")
        genInput = np.random.uniform(-1,1,size=(nEntries,self.noiseDim))
        genLabelInput = np.random.randint(0,self.nClasses, size = (nEntries))
        #genLabelInput = np.array([[1 if i == li else -1 for i in range(self.nClasses)] for li in genLabelInput], dtype='float32')
        genImages = self.generator.predict([genInput, genLabelInput])
        print(self.name + ": finished data generation")
        return genImages, genLabelInput