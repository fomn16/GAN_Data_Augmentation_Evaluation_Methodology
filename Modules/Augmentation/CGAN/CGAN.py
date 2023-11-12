import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Saving import *

from Modules.Datasets.Dataset import Dataset
from Modules.Shared.Params import Params

from Modules.Augmentation.GANFramework import *

class CGAN(GANFramework):
    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4
        self.embeddingDims = 32

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 2.5e-5
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.05
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.ganEpochs = 100
        self.batchSize = 64
        self.extraDiscEpochs = 5
        self.generator = None
        self.discriminator = None
        self.gan = None
        raise ValueError("CGAN.loadConstants must be overriten") 
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 3, 64, 4)
        model = self.TransposedBlock(model, 3, 128, 4)
        model = self.TransposedBlock(model, 3, 256, 3)
        raise ValueError("CGAN.genUpscale must be overriten") 
        return model
    
    def discDownscale(self, model):
        model = self.Block(model, 2, 64, 3)
        model = self.Block(model, 2, 128, 3)
        model = self.Block(model, 2, 256, 3)
        raise ValueError("CGAN.discDownscale must be overriten") 
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
    
    #Cria model geradora com keras functional API
    def createGenModel(self):
        cgenNoiseInput = keras.Input(shape=(self.noiseDim,), name = 'genInput_randomDistribution')

        # Faz reshape para dimensões espaciais desejadas
        cgenX = layers.Reshape((self.genWidth, self.genHeight, self.noiseDepth))(cgenNoiseInput)
    
        labelInput = keras.Input(shape=(1,), name = 'genInput_label')
        embeddedLabels= layers.Embedding(self.nClasses, self.genWidth*self.genHeight*2)(labelInput)
        reshapedLabels = layers.Reshape((self.genWidth, self.genHeight, 2))(embeddedLabels)
        cgenX = layers.concatenate([cgenX, reshapedLabels])

        model = self.genUpscale(cgenX)

        cgenOutput = Conv2D(filters=self.imgChannels, kernel_size=(3,3), padding='same', activation='tanh',  name = 'genOutput_img', kernel_initializer='glorot_uniform')(model)
        
        self.generator = keras.Model(inputs = [cgenNoiseInput, labelInput], outputs = cgenOutput, name = 'cgenerator')

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

        discX = self.discDownscale(discX)

        # nó de output, mapear em -1 ou 1
        discX = Conv2D(1, self.genWidth, kernel_initializer='glorot_uniform', activation='linear')(discX)
        discOutput = Flatten(name = 'discoutput_realvsfake')(discX)

        self.discriminator = keras.Model(inputs = [discInput, labelInput], outputs = discOutput, name = 'discriminator')

        keras.utils.plot_model(
            self.discriminator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/discriminator.png')
        )

    #compilando discriminador e gan
    def compile(self):
        epochPath = self.basePath + '/modelSaves/fold_' + str(self.currentFold) + '/epoch_' + str(loadParam(self.name + '_current_epoch'))
        
        self.createDiscModel()
        self.createGenModel()

        if(self.params.continuing):
            self.discriminator.load_weights(verifiedFolder(epochPath + '/disc_weights'))
            self.generator.load_weights(verifiedFolder(epochPath + '/gen_weights'))
            self.optDiscr = Adam(learning_rate = loadParam(self.name + '_disc_opt_lr'), beta_1=0.5, beta_2=0.9)
            self.optGan  = Adam(learning_rate = loadParam(self.name + '_gan_opt_lr'),  beta_1=0.5, beta_2=0.9)
        else:
            self.optDiscr = Adam(learning_rate = self.initLr, beta_1=0.5, beta_2=0.9)
            self.optGan  = Adam(learning_rate = self.initLr, beta_1=0.5, beta_2=0.9)

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optDiscr, metrics=[my_distance, my_accuracy])

        self.discriminator.trainable = False
        cganNoiseInput = Input(shape=(self.noiseDim,))
        cganLabelInput = Input(shape=(1,))
        cganOutput =  self.discriminator([self.generator([cganNoiseInput, cganLabelInput]), cganLabelInput])
        self.gan = Model((cganNoiseInput, cganLabelInput), cganOutput)

        self.gan.compile(loss='binary_crossentropy', optimizer=self.optGan)
        
        self.discriminator.trainable = True

        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )

        if(not self.params.continuing):
            self.saveModel()

    #treinamento GAN
    def train(self, dataset: Dataset):
        print('started ' + self.name + ' training')
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
            #noise e labels de benchmark
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

            #benchLabels = np.array([[1 if i == bl else -1 for i in range(self.nClasses)] for bl in benchLabels], dtype='float32')
        
        nBatches = int(dataset.trainInstances/self.batchSize) - self.extraDiscEpochs

        for epoch in range(startEpoch+1, self.ganEpochs):
            if(loadParam('close') == True):
                saveParam('close', False)
                self.saveModel(epoch-1, genLossHist, discLossHist)
                sys.exit()
            for i in range(nBatches):
                for j in range(self.extraDiscEpochs):
                    imgBatch, labelBatch = dataset.getTrainData((i+j)*self.batchSize, (i+j+1)*self.batchSize)
                    
                    genInput = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                    labelInput = np.random.randint(0,self.nClasses, size = (self.batchSize))
                    #labelInput = np.array([[1 if i == li else -1 for i in range(self.nClasses)] for li in labelInput], dtype='float32')
                    
                    genImgOutput = self.generator.predict([genInput, labelInput], verbose=0)

                    XImg = np.concatenate((imgBatch, genImgOutput))
                    XLabel = np.concatenate((labelBatch, labelInput))
                    y = ([-1] * self.batchSize) + ([1] * self.batchSize)
                    y = np.reshape(y, (-1,))
                    (XImg, XLabel, y) = shuffle(XImg, XLabel, y)
                    discLoss = self.discriminator.train_on_batch([XImg,XLabel], y)
                
                genTrainNoise = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                genTrainClasses = np.random.randint(0,self.nClasses, size = (self.batchSize))

                gentrainLbls = [-1]*(self.batchSize)
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
                    showOutputAsImg(out, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a) for a in benchLabels[:20]]) + '.png', colored=(self.imgChannels>1))
                    
                    plotLoss([[genLossHist, 'generator loss'],[discLossHist, 'discriminator loss']], self.basePath + '/trainPlot.png')

            if((self.params.saveModels and epoch%5 == 0) or epoch == self.ganEpochs-1):
                self.saveModel(epoch, genLossHist, discLossHist)
                
    #Gera e salva imagens
    def saveGenerationExample(self, nEntries = 20):
        noise = np.random.uniform(-1,1, size=(5*self.nClasses,self.noiseDim))
        labels = np.floor(np.array(range(5*self.nClasses))/5)
        images = self.generator.predict([noise, labels])
        out = ((images * 127.5) + 127.5).astype('uint8')
        showOutputAsImg(out, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a) for a in labels]) + '.png',self.nClasses*5, colored=(self.imgChannels>1))

    def generate(self, srcImgs, srcLbls):
        nEntries = srcLbls.shape[0]
        print(self.name + ": started data generation")
        genInput = np.random.uniform(-1,1,size=(nEntries,self.noiseDim))
        genLabelInput = np.random.randint(0,self.nClasses, size = (nEntries))

        if(self.generator is None):
            self.compile()
        genImages = self.generator.predict([genInput, genLabelInput])
        print(self.name + ": finished data generation")
        return genImages, genLabelInput