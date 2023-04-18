import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

class GAN_CIFAR_10(Augmentator):
    #Constantes:
    genWidth = 4
    genHeight = 4
    genDepth = 128

    noiseDim = 100
    genFCOutputDim = 1024
    discFCOutputDim = 2048

    initLr = 2e-4
    leakyReluAlpha = 0.2
    l2RegParam = 0.01
    dropoutParam = 0.05

    ganEpochs = 500
    batchSize = 128

    generator = None
    discriminator = None
    gan = None

    def __init__(self, params: Params, extraParams = None, nameComplement = ""):
        self.name = self.__class__.__name__ + nameComplement

        self.currentFold = params.currentFold
        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight

        self.params = params

    def AddBlock(self, inModel, nLayers: int, outDepth: int):
        for i in range(nLayers):
            if i == 0:
                model = layers.Conv2D(filters=outDepth, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(self.l2RegParam))(inModel)
            else:
                 model = layers.Conv2D(filters=outDepth, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(self.l2RegParam))(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
            model = layers.Dropout(self.dropoutParam)(model)
        model = layers.MaxPool2D(pool_size=(2,2), padding='valid', strides=(2,2))(model)
        return model
    
    def AddBlockTranspose(self, inModel, nLayers: int, outDepth: int):
        for i in range(nLayers):
            if i == 0:
                model = layers.Conv2DTranspose(filters=outDepth, kernel_size=(3,3), padding='same', strides=(2,2), kernel_regularizer=regularizers.l2(self.l2RegParam))(inModel)
            else:
                model = layers.Conv2DTranspose(filters=outDepth, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(self.l2RegParam))(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
            model = layers.Dropout(self.dropoutParam)(model)
        return model
    
    #Cria model geradora com keras functional API
    def createGenModel(self):
        genInput = keras.Input(shape=(self.noiseDim,), name = 'geninput_randomdistribution')

        # cria camada de entrada, com noiseDim entradas, saída de tamanho sCOutputDim, e ativação relu
        # entrada -> tamanho escolhido
        genX = layers.Dense(self.genFCOutputDim)(genInput)
        genX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(genX)
        genX = layers.Dropout(self.dropoutParam)(genX)

        # cria camada que converte saída da primeira camada para o número de nós necessário na entrada
        # das camadas convolucionais
        genX = layers.Dense(units=self.genWidth*self.genHeight*self.genDepth)(genX)
        genX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(genX)
        genX = layers.BatchNormalization()(genX)

        labelOutput = layers.Dense(self.nClasses, activation='tanh', name='genoutput_label')(genX)

        # Faz reshape para dimensões espaciais desejadas
        genX = layers.Reshape((self.genWidth, self.genHeight, self.genDepth))(genX)

        model = self.AddBlockTranspose(genX, 1, 128)

        model = self.AddBlockTranspose(model, 1, 128)

        model = self.AddBlockTranspose(model, 1, 128)

        model = layers.BatchNormalization()(model)

        # camada convolucional que tem como output a imagem de saída
        # tanh é usado pois é necessária saída de espaço limitado
        genOutput = layers.Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='tanh',  name = 'genOutput_img', kernel_regularizer=regularizers.l2(self.l2RegParam))(model)

        self.generator = keras.Model(inputs = genInput, outputs = [genOutput, labelOutput], name = 'generator')
        
        keras.utils.plot_model(
            self.generator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/generator.png')
        )

    #Cria model discriminadora com functional API
    def createDiscModel(self):
        discInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'discinput_img')

        discX = self.AddBlock(discInput, 1, 64)
        discX = self.AddBlock(discX, 1, 128)
        discX = self.AddBlock(discX, 1, 256)
        discX = self.AddBlock(discX, 1, 256)
        discX = layers.BatchNormalization(axis=-1)(discX)

        # camada densa
        discX = layers.Flatten()(discX)

        discX = layers.Dense(self.discFCOutputDim, activation="tanh")(discX)
        discX = layers.Dropout(self.dropoutParam)(discX)

        labelInput = keras.Input(shape=(self.nClasses,), name = 'discinput_label')
        discX = layers.concatenate([discX, labelInput])

        discX = layers.Dense(self.discFCOutputDim)(discX)
        discX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(discX)
        discX = layers.Dropout(self.dropoutParam)(discX)
        discX = layers.BatchNormalization(axis=-1)(discX)

        # nó de output, sigmoid->mapear em 0 ou 1
        discOutput = layers.Dense(1, activation='sigmoid', name = 'discoutput_realvsfake')(discX)

        self.discriminator = keras.Model(inputs = [discInput, labelInput], outputs = discOutput, name = 'discriminator')

        keras.utils.plot_model(
            self.discriminator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/discriminator.png')
        )

    #compilando discriminador e gan
    def compile(self):
        optDiscr = Adam(learning_rate = self.initLr/2, beta_1 = 0.5)
        self.createDiscModel()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optDiscr)

        self.createGenModel()
        self.discriminator.trainable = False
        input = Input(shape=(self.noiseDim,))
        output = self.discriminator(self.generator(input))
        self.gan = Model(input, output)

        optGan = Adam(learning_rate=self.initLr, beta_1=0.5)
        self.gan.compile(loss= 'binary_crossentropy', optimizer=optGan)

        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )

    #treinamento GAN
    def train(self, dataset:Dataset):
        benchNoise = np.random.uniform(-1,1, size=(256,self.noiseDim))
        genLossHist = []
        discLossHist = []

        imgs, lbls = dataset.getAllTrainData()
        for epoch in range(self.ganEpochs):
            nBatches = int(dataset.trainInstances/self.batchSize)
            for i in range(nBatches):
                imgBatch = imgs[i*self.batchSize:(i+1)*self.batchSize]
                labelBatch = lbls[i*self.batchSize:(i+1)*self.batchSize]
                genInput = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                genImgOutput, genLabelOutput = self.generator.predict(genInput, verbose=0)

                XImg = np.concatenate((imgBatch, genImgOutput))
                XLabel = np.concatenate((labelBatch, genLabelOutput))
                y = ([1] * self.batchSize) + ([0] * self.batchSize)
                y = np.reshape(y, (-1,))
                (XImg, XLabel, y) = shuffle(XImg, XLabel, y)
                
                discLoss = self.discriminator.train_on_batch([XImg,XLabel], y)
                
                genTrainNoise = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                gentrainLbls = [1]*self.batchSize
                gentrainLbls = np.reshape(gentrainLbls, (-1,))
                ganLoss = self.gan.train_on_batch(genTrainNoise,gentrainLbls)
                if i == nBatches-1:
                    discLossHist.append(discLoss)
                    genLossHist.append(ganLoss)
                    print("Epoch " + str(epoch) + "\nGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss))
                    infoFile = open(self.basePath + '/info.txt', 'a')
                    infoFile.write("Epoch " + str(epoch) + "\nGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss) + '\n')
                    infoFile.close()

                    images, labels = self.generator.predict(benchNoise)
                    out = ((images * 127.5) + 127.5).astype('uint8')
                    showOutputAsImg(out, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a.argmax()) for a in labels[:20]]) + '.png', colored=True)
                    plotLoss([[genLossHist, 'generator loss'],[discLossHist, 'discriminator loss']], self.basePath + '/trainPlot.png')
            if(epoch%5 == 0 or epoch == self.ganEpochs-1):
                epochPath = self.basePath + '/modelSaves/fold_' + str(self.currentFold) + '/epoch_' + str(epoch)
                self.generator.save(verifiedFolder(epochPath + '/gen'))
                self.discriminator.save(verifiedFolder(epochPath + '/disc'))
                self.gan.save(verifiedFolder(epochPath + '/gan'))

    #Gera e salva imagens
    def saveGenerationExample(self, nEntries=20):
        noise = np.random.uniform(-1,1, size=(nEntries,self.noiseDim))
        images, labels = self.generator.predict(noise)
        out = ((images * 127.5) + 127.5).astype('uint8')
        showOutputAsImg(out, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a.argmax()) for a in labels]) + '.png',nEntries, colored=True)

    def generate(self, nEntries):
        print(self.name + ": started data generation")
        genInput = np.random.uniform(-1,1,size=(nEntries,self.noiseDim))
        genImg, genLbl = self.generator.predict(genInput, verbose=0)
        print(self.name + ": finished data generation")
        return np.array(genImg[:nEntries]), np.array(genLbl[:nEntries])