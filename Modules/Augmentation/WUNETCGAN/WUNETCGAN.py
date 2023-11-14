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

        self.uNetChannels = 32
        self.uNetRatio = 1.5
        self.uNetBlocks = 3
        
        raise ValueError("WUNETCGAN.loadConstants must be overriten") 
    
    def genUpscale(self, model):
        model = self.TransposedBlock(model, 2, 32)
        model = self.TransposedBlock(model, 2, 32)
        model = self.TransposedBlock(model, 2, 6)
        raise ValueError("WUNETCGAN.genUpscale must be overriten") 
        return model
    
    def discDownscale(self, model):
        model = self.InceptionBlock(model, 2, 32, stride=2)
        model = self.InceptionBlock(model, 2, 32, stride=2)
        model = self.InceptionBlock(model, 2, 32, stride=2)
        raise ValueError("WUNETCGAN.discDownscale must be overriten") 
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
        noise = keras.Input(shape=(self.noiseDim,), name = 'gen_input_gaussian_distribution')
        # Faz reshape para dimensões espaciais desejadas
        X = layers.Reshape((self.genWidth, self.genHeight, self.noiseDepth))(noise)
    
        label = keras.Input(shape=(1,), name = 'gen_input_label')
        embeddedLabels= layers.Embedding(self.nClasses, self.genWidth*self.genHeight)(label)
        reshapedLabels = layers.Reshape((self.genWidth, self.genHeight, 1))(embeddedLabels)
        X = layers.concatenate([X, reshapedLabels])

        X = self.genUpscale(X)

        imageInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'gen_input_img')

        X = layers.concatenate([X, imageInput])

        X = self.UNet(X, self.uNetChannels, self.uNetRatio, self.uNetBlocks)

        output = Conv2D(filters=self.imgChannels, kernel_size=1, padding='same', activation='tanh',  name = 'gen_output', kernel_initializer='glorot_uniform')(X)
        
        self.generator = keras.Model(inputs = [noise, label, imageInput], outputs = output, name = 'generator')
        self.generator.summary()
        keras.utils.plot_model(
            self.generator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/generator.png')
        )
    
    #Cria model discriminadora com functional API
    def createDiscModel(self):
        img1 = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'disc_input_img_1')
        img2 = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'disc_input_img_2')

        X = layers.concatenate([img1, img2])

        X = self.discDownscale(X)

        X = Conv2D(1 + self.nClasses, self.genWidth, kernel_initializer='glorot_uniform', activation='softmax')(X)
        discOutput = Flatten(name = 'discoutput_realvsfake')(X)

        self.discriminator = keras.Model(inputs = [img1, img2], outputs = discOutput, name = 'discriminator')
        self.discriminator.summary()
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
            self.optDiscr = Adam(learning_rate=loadParam(self.name + '_disc_opt_lr'), beta_1 = 0.5, beta_2=0.9)#Adam(learning_rate = self.initLr, beta_1 = 0.5, beta_2=0.9)
            self.optGan = Adam(learning_rate=loadParam(self.name + '_gan_opt_lr'), beta_1 = 0.5, beta_2=0.9)#Adam(learning_rate = self.initLr*10, beta_1=0.5, beta_2=0.9)
        else:
            self.optDiscr = Adam(learning_rate=self.initLr, beta_1 = 0.5, beta_2=0.9)#Adam(learning_rate = self.initLr, beta_1 = 0.5, beta_2=0.9)
            self.optGan = Adam(learning_rate=self.initLr, beta_1 = 0.5, beta_2=0.9)#Adam(learning_rate = self.initLr*10, beta_1=0.5, beta_2=0.9)

        self.discriminator.compile(loss=wasserstein_loss, optimizer=self.optDiscr, metrics=[my_distance, 'accuracy'])

        self.discriminator.trainable = False
        cganNoiseInput = Input(shape=(self.noiseDim,))
        cganLabelInput = Input(shape=(1,))
        cganImgInput = Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels))
        cganOutput =  self.discriminator([self.generator([cganNoiseInput, cganLabelInput, cganImgInput]), cganImgInput])
        self.gan = Model((cganNoiseInput, cganLabelInput, cganImgInput), cganOutput)

        self.gan.compile(loss=wasserstein_loss, optimizer=self.optGan)
        
        self.discriminator.trainable = True
        self.gan.summary()
        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )

        if(not self.params.continuing):
            self.saveModel()

    #treinamento GAN
    def train(self, dataset: Dataset):
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
            #noise e labels de benchmark
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

                    imgsShuffled = shuffle_same_class(imgBatch, labelBatch, self.nClasses)
                    imgsWrongClass, lblsWrongClass = shuffle_different_class(imgBatch, labelBatch, self.nClasses)

                    genInput = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                    imgsFake = self.generator.predict([genInput, labelBatch, imgBatch], verbose=0)
                    
                    XImg    = np.concatenate((imgsShuffled, imgsFake, imgsWrongClass))
                    XImg2   = np.concatenate((imgBatch, imgBatch, imgBatch))


                    fake = [0] * self.nClasses + [1]
                    fake = [fake for _ in range(self.batchSize)]

                    real = [[1 if i == c else 0 for c in range(self.nClasses + 1)] for i in labelBatch]
                
                    y = np.array((real) + (fake) + (fake))

                    (XImg, XImg2, y) = shuffle(XImg, XImg2, y)
                    discLoss = self.discriminator.train_on_batch([XImg, XImg2], y)

                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                        l.set_weights(weights)
                
                imgBatch, labelBatch = dataset.getTrainData((i)*self.batchSize, (i+1)*self.batchSize)
                genTrainNoise = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                y = np.array([[1 if i == c else 0 for c in range(self.nClasses + 1)] for i in labelBatch])

                ganLoss = self.gan.train_on_batch([genTrainNoise, labelBatch, imgBatch],y)

                if i == nBatches-1:
                    discLossHist.append(discLoss)
                    genLossHist.append(ganLoss)

                    print("Epoch " + str(epoch) + "\nCGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss))
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
                
    #Gera e salva imagens
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
        if(self.generator is None):
            self.compile()
        genImages = self.generator.predict([noise, srcLbls, srcImgs])
        print(self.name + ": finished data generation")
        return genImages, srcLbls