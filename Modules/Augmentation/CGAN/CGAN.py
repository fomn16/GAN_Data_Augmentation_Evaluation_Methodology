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

        self.ganEpochs = 75
        self.batchSize = 128
        self.extraDiscEpochs = 5
        self.generator = None
        self.discriminator = None
        self.gan = None
        raise ValueError("CGAN.loadConstants must be overriten") 
    
    def genInputProcessing(self, noise, label):
        X = layers.Reshape((self.genWidth, self.genHeight, self.noiseDepth))(noise)
        XLbl = layers.Embedding(self.nClasses, self.genWidth*self.genHeight)(label)
        XLbl = layers.Reshape((self.genWidth, self.genHeight, 1))(XLbl)

        X = layers.concatenate([X, XLbl])
        return X
        
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 3, 64, 4)
        model = self.TransposedBlock(model, 3, 128, 4)
        model = self.TransposedBlock(model, 3, 256, 3)
        raise ValueError("CGAN.genUpscale must be overriten") 
        return model

    def discInputProcessing(self, image, label):
        X = layers.Embedding(self.nClasses, self.imgWidth*self.imgHeight)(label)
        X = layers.Reshape((self.imgWidth, self.imgHeight, 1))(X)
        X = layers.concatenate([image, X])
        return X

    def discDownscale(self, model):
        model = self.Block(model, 2, 64, 3)
        model = self.Block(model, 2, 128, 3)
        model = self.Block(model, 2, 256, 3)
        raise ValueError("CGAN.discDownscale must be overriten") 
        return model
    
    def discOutputProcessing(self, model):
        X = Conv2D(1, self.genWidth, kernel_initializer='glorot_uniform', activation='linear')(model)
        X = Flatten(name = 'discoutput_realvsfake')(X)
        return X

    def __init__(self, params: Params, extraParams = None, nameComplement = ""):
        self.name = self.__class__.__name__ + addToName("(" +  nameComplement + ")")

        self.currentFold = params.currentFold
        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight

        self.params = params

        self.loadConstants()
    
    def createGenModel(self):
        noise = keras.Input(shape=(self.noiseDim,), name = 'genInput_randomDistribution')
        label = keras.Input(shape=(1,), name = 'genInput_label')

        X = self.genInputProcessing(noise, label)

        X = self.genUpscale(X)

        cgenOutput = Conv2D(filters=self.imgChannels, kernel_size=(3,3), padding='same', activation='tanh',  name = 'genOutput_img', kernel_initializer='glorot_uniform')(X)
        
        self.generator = keras.Model(inputs = [noise, label], outputs = cgenOutput, name = 'cgenerator')

        self.generator.summary()
        keras.utils.plot_model(
            self.generator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/generator.png')
        )

    def createDiscModel(self):
        image = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'disc_image')
        label = keras.Input(shape=(1,), name = 'disc_label')

        X = self.discInputProcessing(image, label)

        X = self.discDownscale(X)

        X = self.discOutputProcessing(X)

        self.discriminator = keras.Model(inputs = [image, label], outputs = X, name = 'discriminator')

        self.discriminator.summary()
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
            
        nBatches = int(self.params.datasetTrainInstances/self.batchSize) - self.extraDiscEpochs
        lr_schedule_disc = ExponentialDecay(
            self.initLr, 
            staircase = False, 
            decay_steps=self.ganEpochs*self.extraDiscEpochs*nBatches,
            decay_rate=0.96
        )
        lr_schedule_gan = ExponentialDecay(
            self.initLr, 
            staircase = False, 
            decay_steps=self.ganEpochs*nBatches, 
            decay_rate=0.96
        )
            
        self.optDiscr = Adam(learning_rate = lr_schedule_disc, beta_1=0.5, beta_2=0.9)
        self.optGan  = Adam(learning_rate = lr_schedule_gan, beta_1=0.5, beta_2=0.9)

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optDiscr, metrics=[my_accuracy])

        self.discriminator.trainable = False
        cganNoiseInput = Input(shape=(self.noiseDim,))
        cganLabelInput = Input(shape=(1,))
        cganOutput =  self.discriminator([self.generator([cganNoiseInput, cganLabelInput]), cganLabelInput])
        self.gan = Model((cganNoiseInput, cganLabelInput), cganOutput)

        self.gan.compile(loss='binary_crossentropy', optimizer=self.optGan)
        
        self.discriminator.trainable = True

        self.gan.summary()
        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )

        if(not self.params.continuing):
            self.saveModel()

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

                    print("Epoch " + str(epoch) + "\ngenerator loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss))
                    infoFile = open(self.basePath + '/info.txt', 'a')
                    infoFile.write("Epoch " + str(epoch) + "\ngenerator training loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss)+ '\n')
                    infoFile.close()

                    images = self.generator.predict([benchNoise, benchLabels])
                    out = ((images * 127.5) + 127.5).astype('uint8')
                    showOutputAsImg(out, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a) for a in benchLabels[:20]]) + '.png', colored=(self.imgChannels>1))
                    
                    plotLoss([[genLossHist, 'generator loss'],[discLossHist, 'discriminator loss']], self.basePath + '/trainPlot.png')

            if((self.params.saveModels and epoch%5 == 0) or epoch == self.ganEpochs-1):
                self.saveModel(epoch, genLossHist, discLossHist)
                
    def saveGenerationExample(self, nEntries = 20):
        noise = np.random.uniform(-1,1, size=(5*self.nClasses,self.noiseDim))
        labels = np.floor(np.array(range(5*self.nClasses))/5)
        images = self.generator.predict([noise, labels])
        out = ((images * 127.5) + 127.5).astype('uint8')
        if(self.nClasses <= 10):
            filepath = self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a) for a in labels]) + '.png'
        else:
            filepath = self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a) for a in labels[:50]]) + '.png'
        showOutputAsImg(out, filepath, self.nClasses*5, colored=(self.imgChannels>1))

    def generate(self, srcImgs, srcLbls):
        nEntries = srcLbls.shape[0]
        print(self.name + ": started data generation")
        genInput = np.random.uniform(-1,1,size=(nEntries,self.noiseDim))
        genLabelInput = np.random.randint(0,self.nClasses, size = (nEntries))

        genImages = self.generator.predict([genInput, genLabelInput])
        print(self.name + ": finished data generation")
        return genImages, genLabelInput