import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

from Modules.Datasets.Dataset import Dataset
from Modules.Shared.Params import Params

from Modules.Augmentation.GANFramework import *

class WUNETCGAN(GANFramework):
    def loadConstants(self):
        self.genWidth = 4
        self.genHeight = 4

        approximateNoiseDim = 100
        self.noiseDepth = int(np.ceil(approximateNoiseDim/(self.genWidth*self.genHeight)))
        self.noiseDim = self.genWidth*self.genHeight*self.noiseDepth

        self.initLr = 2e-5
        self.leakyReluAlpha = 0.2
        self.dropoutParam = 0.02
        self.batchNormMomentum = 0.8
        self.batchNormEpsilon = 2e-4

        self.clipValue = 0.01

        self.ganEpochs = 100
        self.batchSize = 64
        self.extraDiscEpochs = 2
        self.generator = None
        self.discriminator = None
        self.gan = None

        self.wrongClassAmmt = 0.25
        
        raise ValueError("WUNETCGAN.loadConstants must be overriten") 
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 2, 16, dropout=False)
        model = self.TransposedBlock(model, 2, 32, dropout=False)
        model = self.TransposedBlock(model, 2, 8, dropout=False)
        raise ValueError("WUNETCGAN.genUpscale must be overriten") 
        return model
    
    def discDownscale(self, model):
        model = self.InceptionBlock(model, 3, 32, stride=2, dropout=False)
        model = self.InceptionBlock(model, 3, 64, stride=2, dropout=False)
        model = self.InceptionBlock(model, 3, 128, stride=2, dropout=False)
        raise ValueError("WUNETCGAN.discDownscale must be overriten") 
        return model

    def embeddingProcessing(self, model):
        ret = layers.Embedding(self.nClasses, self.genWidth*self.genHeight)(model)
        ret = layers.Reshape((self.genWidth, self.genHeight, 1))(ret)
        return ret
    
    def UNetCall(self, model):
        return self.UNet(model, 32, 1.5, 3, dropout=False, batchNorm=True)

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
        noise = keras.Input(shape=(self.noiseDim,), name = 'gen_input_gaussian_distribution')
        X = layers.Reshape((self.genWidth, self.genHeight, self.noiseDepth))(noise)
    
        label = keras.Input(shape=(1,), name = 'gen_input_label')
        reshapedLabels = self.embeddingProcessing(label)
        
        X = layers.concatenate([X, reshapedLabels])

        X = self.genUpscale(X)

        imageInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'gen_input_img')

        X = layers.concatenate([X, imageInput])

        X = self.UNetCall(X)

        output = Conv2D(filters=self.imgChannels, kernel_size=1, padding='same', activation='tanh',  name = 'gen_output', kernel_initializer='glorot_uniform')(X)
        
        self.generator = keras.Model(inputs = [noise, label, imageInput], outputs = output, name = 'generator')
        
        self.generator.summary()
        keras.utils.plot_model(
            self.generator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/generator.png')
        )
    
    def createDiscModel(self):
        img1 = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'disc_input_img_1')
        img2 = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'disc_input_img_2')

        X = layers.concatenate([img1, img2])

        X = self.discDownscale(X)

        X1 = Conv2D(self.nClasses, self.genWidth, kernel_initializer='glorot_uniform', activation='softmax')(X)
        discOutput1 = Flatten(name = 'discoutput_class')(X1)

        X2 = Conv2D(1, self.genWidth, kernel_initializer='glorot_uniform', activation='tanh')(X)
        discOutput2 = Flatten(name = 'discoutput_flag')(X2)

        self.discriminator = keras.Model(inputs = [img1, img2], outputs = [discOutput1, discOutput2], name = 'discriminator')
        
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
        lr_schedule_gen = ExponentialDecay(
            self.initLr/2, 
            staircase = False,
            decay_steps=self.ganEpochs*nBatches/10,
            decay_rate=0.93
        
        )
        self.optDiscr = Adam(learning_rate=lr_schedule_disc, beta_1 = 0.5, beta_2=0.9)
        self.optGan =   Adam(learning_rate=lr_schedule_gan, beta_1 = 0.5, beta_2=0.9)
        self.optGen =   Adam(learning_rate=lr_schedule_gen, beta_1 = 0.5, beta_2=0.9)

        self.discriminator.compile( loss=['categorical_crossentropy',wasserstein_loss], 
                                    optimizer=self.optDiscr, 
                                    metrics=[my_accuracy],
                                    #loss_weights=[1, self.nClasses]
                                   )
        self.generator.compile(loss=tf.losses.mean_squared_error, 
                         optimizer=self.optGen
        )

        self.discriminator.trainable = False
        cganNoiseInput = Input(shape=(self.noiseDim,))
        cganLabelInput = Input(shape=(1,))
        cganImgInput = Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels))
        cganOutput =  self.discriminator([self.generator([cganNoiseInput, cganLabelInput, cganImgInput]), cganImgInput])
        self.gan = Model((cganNoiseInput, cganLabelInput, cganImgInput), cganOutput)

        self.gan.compile(   loss=['categorical_crossentropy',wasserstein_loss],
                            optimizer=self.optGan,
                            #loss_weights=[1, self.nClasses]
                        )
        
        self.discriminator.trainable = True
        
        self.gan.summary()
        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )

        if(not self.params.continuing):
            self.saveModel()

    def train(self, dataset: Dataset):
        self.ld = {}
        print('started ' + self.name + ' training')
        self.testImgs, self.testLbls = dataset.getTestData(0, 20)

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
            benchNoise = np.random.uniform(-1,1, size=(self.batchSize,self.noiseDim))
            benchLabels = np.random.randint(0,self.nClasses, size = (self.batchSize))
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
                    (imgBatch, labelBatch) = shuffle(imgBatch, labelBatch)
                    
                    nReal = int(self.batchSize/2)
                    nWrong = int((self.batchSize - nReal)*self.wrongClassAmmt)
                    nFake = self.batchSize - nWrong
                    
                    imgBatch = [imgBatch[:nReal], imgBatch[nReal:nFake], imgBatch[nFake:]]
                    labelBatch = [labelBatch[:nReal], labelBatch[nReal:nFake], labelBatch[nFake:]]

                    imgsShuffled = shuffle_same_class(imgBatch[0], labelBatch[0], self.nClasses)

                    genInput = np.random.uniform(-1,1,size=(len(labelBatch[1]),self.noiseDim))
                    imgsFake = self.generator.predict([genInput, labelBatch[1], imgBatch[1]], verbose=0)

                    imgsWrongClass, _ = shuffle_different_class(imgBatch[2], labelBatch[2], self.nClasses)
                    
                    XImg    = np.concatenate((imgsShuffled, imgsFake,       imgBatch[2]))
                    XImg2   = np.concatenate((imgBatch[0],  imgBatch[1],    imgsWrongClass))

                    classes = [
                        [[1 if i == c else 0 for c in range(self.nClasses)] for i in labelBatch[0]],
                        [[1 if i == c else 0 for c in range(self.nClasses)] for i in labelBatch[1]],
                        [[1 if i == c else 0 for c in range(self.nClasses)] for i in labelBatch[2]]
                    ]
                    y1 = np.array((classes[0]) + (classes[1]) + (classes[2]))

                    flags = [
                        [-1 for i in labelBatch[0]],
                        [1 for i in labelBatch[1]],
                        [1 for i in labelBatch[2]],
                    ]
                    y2 = np.array((flags[0]) + (flags[1]) + (flags[2]))

                    (XImg, XImg2, y1, y2) = shuffle(XImg, XImg2, y1, y2)
                    discLoss = self.discriminator.train_on_batch([XImg, XImg2], [y1, y2])

                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                        l.set_weights(weights)
                
                imgBatch, labelBatch = dataset.getTrainData((i)*self.batchSize, (i+1)*self.batchSize)
                genTrainNoise = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))

                y1 = np.array([[1 if i == c else 0 for c in range(self.nClasses)] for i in labelBatch])
                y2 = np.array([-1 for i in labelBatch])

                genLoss = self.generator.train_on_batch([genTrainNoise, labelBatch, imgBatch], imgBatch)
                ganLoss = self.gan.train_on_batch([genTrainNoise, labelBatch, imgBatch], [y1, y2])

                if i == nBatches-1:
                    discLossHist.append(discLoss)
                    genLossHist.append(ganLoss)

                    print("Epoch " + str(epoch) + "\ngenerator adversarial loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss))
                    infoFile = open(self.basePath + '/info.txt', 'a')
                    infoFile.write("Epoch " + str(epoch) + "\nCGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss)+ '\n')
                    infoFile.close()

                    images = self.generator.predict([benchNoise[:20], labelBatch[:20], imgBatch[:20]])
                    out = ((images * 127.5) + 127.5).astype('uint8')
                    imagesDs = ((imgBatch[:20] * 127.5) + 127.5).astype('uint8')
                    newOut = np.ndarray(out.shape, out.dtype)
                    for outId in range(10):
                        newOut[outId*2] = imagesDs[outId]
                        newOut[outId*2+1] = out[outId]
                    showOutputAsImg(newOut, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a) for a in labelBatch[:10]]) + '.png', colored=(self.imgChannels>1))
                    plotLoss([[genLossHist, 'generator loss'],[discLossHist, 'discriminator loss']], self.basePath + '/trainPlot.png')
                    
            if(self.params.saveModels):
                self.saveModel(epoch, genLossHist, discLossHist)
                
    def saveGenerationExample(self, nEntries = 20):
        benchNoise = np.random.uniform(-1,1, size=(20,self.noiseDim))
        images = self.generator.predict([benchNoise, self.testLbls, self.testImgs])
        out = ((images * 127.5) + 127.5).astype('uint8')
        imagesDs = ((self.testImgs * 127.5) + 127.5).astype('uint8')
        newOut = np.ndarray(out.shape, out.dtype)
        for outId in range(10):
            newOut[outId*2] = imagesDs[outId]
            newOut[outId*2+1] = out[outId]

        showOutputAsImg(newOut, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a) for a in self.testLbls]) + '.png', colored=(self.imgChannels>1))

    def generate(self, srcImgs, srcLbls):
        nEntries = srcLbls.shape[0]
        print(self.name + ": started data generation")
        noise = np.random.uniform(-1,1,size=(nEntries,self.noiseDim))
        genImages = self.generator.predict([noise, srcLbls, srcImgs])
        print(self.name + ": finished data generation")
        return genImages, srcLbls.copy()