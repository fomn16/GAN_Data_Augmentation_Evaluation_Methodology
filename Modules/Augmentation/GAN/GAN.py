import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Saving import *

from Modules.Datasets.Dataset import Dataset
from Modules.Shared.Params import Params

from Modules.Augmentation.GANFramework import GANFramework

class GAN(GANFramework):
    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4
        self.genDepth = 128

        self.noiseDim = 100
        self.genFCOutputDim = 1024
        self.discFCOutputDim = 2048

        self.initLr = 2e-4
        self.leakyReluAlpha = 0.2
        self.l2RegParam = 0.01
        self.dropoutParam = 0.05
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.ganEpochs = 100
        self.batchSize = 64

        self.generator = None
        self.discriminator = None
        self.gan = None
        raise ValueError("GAN.loadConstants must be overriten") 
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 1, 128, kernel_regularizer=regularizers.l2(self.l2RegParam))
        model = self.TransposedBlock(model, 1, 128, kernel_regularizer=regularizers.l2(self.l2RegParam))
        model = self.TransposedBlock(model, 1, 128, kernel_regularizer=regularizers.l2(self.l2RegParam))
        raise ValueError("GAN.genUpscale must be overriten") 
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 1, 64, kernel_regularizer=regularizers.l2(self.l2RegParam))
        model = self.Block(model, 1, 128, kernel_regularizer=regularizers.l2(self.l2RegParam))
        model = self.Block(model, 1, 256, kernel_regularizer=regularizers.l2(self.l2RegParam))
        model = self.Block(model, 1, 256, kernel_regularizer=regularizers.l2(self.l2RegParam))
        raise ValueError("GAN.discDownscale must be overriten") 
        return model

    def __init__(self, params: Params, extraParams = None, nameComplement = ""):
        self.name = self.__class__.__name__ + "_" +  nameComplement

        self.currentFold = params.currentFold
        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight

        self.params = params

        self.loadConstants()

    def createGenModel(self):
        genInput = keras.Input(shape=(self.noiseDim,), name = 'geninput_randomdistribution')

        genX = layers.Dense(self.genFCOutputDim)(genInput)
        genX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(genX)
        genX = layers.Dropout(self.dropoutParam)(genX)

        genX = layers.Dense(units=self.genWidth*self.genHeight*self.genDepth)(genX)
        genX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(genX)
        genX = layers.BatchNormalization()(genX)

        labelOutput = layers.Dense(self.nClasses, activation='tanh', name='genoutput_label')(genX)

        genX = layers.Reshape((self.genWidth, self.genHeight, self.genDepth))(genX)
        model = self.genUpscale(genX)

        genOutput = layers.Conv2D(filters=self.imgChannels, kernel_size=(3,3), padding='same', activation='tanh',  name = 'genOutput_img', kernel_regularizer=regularizers.l2(self.l2RegParam))(model)

        self.generator = keras.Model(inputs = genInput, outputs = [genOutput, labelOutput], name = 'generator')
        
        keras.utils.plot_model(
            self.generator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/generator.png')
        )

    def createDiscModel(self):
        discInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'discinput_img')

        discX = self.discDownscale(discInput)

        discX = layers.BatchNormalization(axis=-1)(discX)

        discX = layers.Flatten()(discX)

        discX = layers.Dense(self.discFCOutputDim, activation="tanh")(discX)
        discX = layers.Dropout(self.dropoutParam)(discX)

        labelInput = keras.Input(shape=(self.nClasses,), name = 'discinput_label')
        discX = layers.concatenate([discX, labelInput])

        discX = layers.Dense(self.discFCOutputDim)(discX)
        discX = layers.LeakyReLU(alpha=self.leakyReluAlpha)(discX)
        discX = layers.Dropout(self.dropoutParam)(discX)

        discOutput = layers.Dense(1, activation='sigmoid', name = 'discoutput_realvsfake')(discX)

        self.discriminator = keras.Model(inputs = [discInput, labelInput], outputs = discOutput, name = 'discriminator')

        keras.utils.plot_model(
            self.discriminator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/discriminator.png')
        )

    def compile(self):
        epochPath = self.basePath + '/modelSaves/fold_' + str(self.currentFold) + '/epoch_' + str(loadParam(self.name + '_current_epoch'))

        
        self.createDiscModel()
        self.createGenModel()

        if(self.params.continuing):
            self.discriminator.load_weights(verifiedFolder(epochPath + '/disc_weights'))
            self.generator.load_weights(verifiedFolder(epochPath + '/gen_weights'))
            self.optDiscr = Adam(learning_rate = loadParam(self.name + '_disc_opt_lr'), beta_1=0.5)
            self.optGan  = Adam(learning_rate = loadParam(self.name + '_gan_opt_lr'),  beta_1=0.5)
        else:
            self.optDiscr = Adam(learning_rate = self.initLr/2, beta_1 = 0.5)
            self.optGan  = Adam(learning_rate=self.initLr, beta_1=0.5)

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optDiscr)

        self.discriminator.trainable = False
        input = Input(shape=(self.noiseDim,))
        output = self.discriminator(self.generator(input))
        self.gan = Model(input, output)

        self.gan.compile(loss= 'binary_crossentropy', optimizer=self.optGan)

        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )

        if(not self.params.continuing):
            self.saveModel()

    def train(self, dataset:Dataset):
        discLossHist = []
        genLossHist = []
        benchNoise = None
        benchLabels = None
        startEpoch = None
        if(self.params.continuing):
            benchNoise = np.array(loadParam(self.name + '_bench_noise'))
            benchLabels = np.array(loadParam(self.name + '_bench_labels'))
            startEpoch = loadParam(self.name + '_current_epoch')
            discLossHist = loadParam(self.name + '_disc_loss_hist')
            genLossHist = loadParam(self.name + '_gen_loss_hist')
        else:
            benchNoise = np.random.uniform(-1,1, size=(256,self.noiseDim))
            benchLabels = np.random.randint(0,self.nClasses, size = (256))
            for i in range(20):
                benchLabels[i] = int(i/2)
            startEpoch = -1
            saveParam(self.name + '_bench_noise', benchNoise.tolist())
            saveParam(self.name + '_bench_labels', benchLabels.tolist())
            saveParam(self.name + '_current_epoch', 0)
            saveParam(self.name + '_disc_loss_hist', [])
            saveParam(self.name + '_gen_loss_hist', [])

        nBatches = int(dataset.trainInstances/self.batchSize)

        for epoch in range(startEpoch+1, self.ganEpochs):
            if(loadParam('close') == True):
                saveParam('close', False)
                self.saveModel(epoch-1, genLossHist, discLossHist)
                sys.exit()
            for i in range(nBatches):
                imgBatch, labelBatch = dataset.getTrainData(i*self.batchSize, (i+1)*self.batchSize)
                labelBatch = np.array([[1 if i == li else -1 for i in range(self.nClasses)] for li in labelBatch], dtype='float32')
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
                    showOutputAsImg(out, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a.argmax()) for a in labels[:20]]) + '.png', colored=(self.imgChannels>1))
                    plotLoss([[genLossHist, 'generator loss'],[discLossHist, 'discriminator loss']], self.basePath + '/trainPlot.png')
                    
            if((self.params.saveModels and epoch%5 == 0) or epoch == self.ganEpochs-1):
                self.saveModel(epoch, genLossHist, discLossHist)

    def saveGenerationExample(self, nEntries=20):
        noise = np.random.uniform(-1,1, size=(nEntries,self.noiseDim))
        images, labels = self.generator.predict(noise)
        out = ((images * 127.5) + 127.5).astype('uint8')
        showOutputAsImg(out, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a.argmax()) for a in labels]) + '.png',nEntries, colored=(self.imgChannels>1))

    def generate(self, srcImgs, srcLbls):
        nEntries = srcLbls.shape[0]
        print(self.name + ": started data generation")
        genInput = np.random.uniform(-1,1,size=(nEntries,self.noiseDim))
        genImg, genLbl = self.generator.predict(genInput, verbose=0)
        genLbl = [a.argmax() for a in genLbl]
        print(self.name + ": finished data generation")
        return np.array(genImg[:nEntries]), np.array(genLbl[:nEntries])