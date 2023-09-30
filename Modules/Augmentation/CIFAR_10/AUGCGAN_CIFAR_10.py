import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Saving import *

def wasserstein_loss(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)#K.cast(y_true, dtype=tf.float32)
    return -K.mean(y_true * y_pred)

def my_distance(y_true, y_pred):
    #input range is [-1,1]
    y_true = tf.cast(y_true, y_pred.dtype)  # Ensure the same data type
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def my_accuracy(y_true, y_pred):
    #input range is [-1,1]
    y_true = tf.cast(y_true, y_pred.dtype)  # Ensure the same data type
    
    # Calculate the signs of y_true and y_pred
    y_true_sign = tf.sign(y_true)
    y_pred_sign = tf.sign(y_pred)
    
    # Calculate the element-wise equality of signs
    sign_equal = tf.equal(y_true_sign, y_pred_sign)
    
    # Calculate the percentage of time with the same sign
    return tf.reduce_mean(tf.cast(sign_equal, tf.float32)) * 100.0

def shuffle_no_repeat(imgs, lbls):
    n = len(imgs)
    shuffledIndices = np.random.permutation(n)
    shuffledImgs = imgs[shuffledIndices]
    shuffledLbls = lbls[shuffledIndices]
    return shuffledImgs, shuffledLbls

class AUGCGAN_CIFAR_10(Augmentator):
    #Constantes:
    genWidth = 4
    genHeight = 4
    embeddingDims = 32

    approximateNoiseDim = 100
    noiseDepth = int(np.ceil(approximateNoiseDim/(genWidth*genHeight)))
    noiseDim = genWidth*genHeight*noiseDepth

    initLr = 2.5e-5
    leakyReluAlpha = 0.2
    dropoutParam = 0.05
    batchNormMomentum = 0.8
    batchNormEpsilon = 2e-4

    clipValue = 0.01

    ganEpochs = 100
    batchSize = 64
    extraDiscEpochs = 5
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
    
    def AddTransposedBlock(self, model, nLayers: int, outDepth: int, kernelSize:int = 3):
        for i in range(nLayers):
            if i == 0:
                model = Conv2DTranspose(filters=outDepth, kernel_size=kernelSize, padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(model)
            else:
                model = Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform')(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)
        return model
    
    def AddResidualBlock(self, model, nLayers:int, outDepth:int, kernelSize:int = 3, stride:int = 1):
        identity = model
        if(stride != 1):
            identity = layers.AveragePooling2D(stride)(identity)
        identity = Conv2D(filters=outDepth, kernel_size=1, padding='same', kernel_initializer='glorot_uniform')(identity)
        identity = layers.LeakyReLU(alpha=self.leakyReluAlpha)(identity)

        for i in range(nLayers):
            model = Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides = (stride if i == 0 else 1))(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            if(i != nLayers - 1):
                model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)

        model = layers.add([model, identity])
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)
        return model
    
    def AddTransposedResidualBlock(self, model, nLayers:int, outDepth:int, kernelSize:int = 3, stride:int = 1):
        identity = model
        if(stride != 1):
            identity = layers.UpSampling2D(stride)(identity)
        identity = Conv2D(filters=outDepth, kernel_size=1, padding='same', kernel_initializer='glorot_uniform')(identity)
        identity = layers.LeakyReLU(alpha=self.leakyReluAlpha)(identity)

        for i in range(nLayers):
            model = Conv2DTranspose(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides = (stride if i == 0 else 1))(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            if(i != nLayers - 1):
                model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)

        model = layers.add([model, identity])
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)
        return model
    
    #Cria model geradora com keras functional API
    def createGenModel(self):
        noise = keras.Input(shape=(self.noiseDim,), name = 'gen_input_gaussian_distribution')
        # Faz reshape para dimensões espaciais desejadas
        X = layers.Reshape((self.genWidth, self.genHeight, self.noiseDepth))(noise)
    
        label = keras.Input(shape=(1,), name = 'gen_input_label')
        embeddedLabels= layers.Embedding(self.nClasses, self.genWidth*self.genHeight*2)(label)
        reshapedLabels = layers.Reshape((self.genWidth, self.genHeight, 2))(embeddedLabels)
        X = layers.concatenate([X, reshapedLabels])

        X = self.AddTransposedBlock(X, 1, 32)
        X = self.AddTransposedBlock(X, 1, 32)
        X = self.AddTransposedBlock(X, 1, 3)

        imageInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'gen_input_img')

        X = layers.concatenate([X, imageInput])
        
        X = self.AddResidualBlock(X, 2, 64, 3)
        X = self.AddResidualBlock(X, 2, 128, 3)
        X = self.AddResidualBlock(X, 2, 256, 3)

        output = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='tanh',  name = 'gen_output', kernel_initializer='glorot_uniform')(X)
        
        self.generator = keras.Model(inputs = [noise, label, imageInput], outputs = output, name = 'generator')

        keras.utils.plot_model(
            self.generator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/generator.png')
        )

    #Cria model discriminadora com functional API
    def createDiscModel(self):
        img1 = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'disc_input_img_1')
        img2 = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'disc_input_img_2')

        label1 = keras.Input(shape=(1,), name = 'disc_input_label_1')
        embeddedLabel1 = layers.Embedding(self.nClasses, self.imgWidth*self.imgHeight)(label1)
        reshapedLabel1 = layers.Reshape((self.imgWidth, self.imgHeight, 1))(embeddedLabel1)
        
        label2 = keras.Input(shape=(1,), name = 'disc_input_label_2')
        embeddedLabel2 = layers.Embedding(self.nClasses, self.imgWidth*self.imgHeight)(label2)
        reshapedLabel2 = layers.Reshape((self.imgWidth, self.imgHeight, 1))(embeddedLabel2)

        X = layers.concatenate([img1, img2, reshapedLabel1, reshapedLabel2])

        X = self.AddResidualBlock(X, 2, 64, 3, 2)
        X = self.AddResidualBlock(X, 2, 128, 3, 2)
        X = self.AddResidualBlock(X, 2, 128, 3, 2)

        X = Conv2D(1, 4, kernel_initializer='glorot_uniform', activation='linear')(X)
        discOutput = Flatten(name = 'discoutput_realvsfake')(X)

        self.discriminator = keras.Model(inputs = [img1, label1, img2, label2], outputs = discOutput, name = 'discriminator')

        keras.utils.plot_model(
            self.discriminator, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/discriminator.png')
        )

    def saveModel(self, epoch = 0, genLossHist = [], discLossHist = []):
        saveParam(self.name + '_current_epoch', epoch)
        saveParam(self.name + '_gen_loss_hist', genLossHist)
        saveParam(self.name + '_disc_loss_hist', discLossHist)
        epochPath = self.basePath + '/modelSaves/fold_' + str(self.currentFold) + '/epoch_' + str(epoch)

        self.discriminator.save_weights(verifiedFolder(epochPath + '/disc_weights'))
        self.generator.save_weights(verifiedFolder(epochPath + '/gen_weights'))

        saveParam(self.name + '_disc_opt_lr', np.float64(self.optDiscr._decayed_lr('float32').numpy()))
        saveParam(self.name + '_gan_opt_lr', np.float64(self.optcGan._decayed_lr('float32').numpy()))

    #compilando discriminador e gan
    def compile(self):
        epochPath = self.basePath + '/modelSaves/fold_' + str(self.currentFold) + '/epoch_' + str(loadParam(self.name + '_current_epoch'))
        
        self.createDiscModel()
        self.createGenModel()

        if(self.params.continuing):
            self.discriminator.load_weights(verifiedFolder(epochPath + '/disc_weights'))
            self.generator.load_weights(verifiedFolder(epochPath + '/gen_weights'))
            self.optDiscr = RMSprop(learning_rate=loadParam(self.name + '_disc_opt_lr'))#Adam(learning_rate = self.initLr, beta_1 = 0.5, beta_2=0.9)
            self.optcGan = RMSprop(learning_rate=loadParam(self.name + '_gan_opt_lr'))#Adam(learning_rate = self.initLr*10, beta_1=0.5, beta_2=0.9)
        else:
            self.optDiscr = RMSprop(learning_rate=self.initLr)#Adam(learning_rate = self.initLr, beta_1 = 0.5, beta_2=0.9)
            self.optcGan = RMSprop(learning_rate=self.initLr)#Adam(learning_rate = self.initLr*10, beta_1=0.5, beta_2=0.9)

        self.discriminator.compile(loss=wasserstein_loss, optimizer=self.optDiscr, metrics=[my_distance, my_accuracy])

        self.discriminator.trainable = False
        cganNoiseInput = Input(shape=(self.noiseDim,))
        cganLabelInput = Input(shape=(1,))
        cganImgInput = Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels))
        cganOutput =  self.discriminator([self.generator([cganNoiseInput, cganLabelInput, cganImgInput]), cganLabelInput, cganImgInput, cganLabelInput])
        self.gan = Model((cganNoiseInput, cganLabelInput, cganImgInput), cganOutput)

        self.gan.compile(loss=wasserstein_loss, optimizer=self.optcGan)
        
        self.discriminator.trainable = True

        keras.utils.plot_model(
            self.gan, show_shapes= True, show_dtype = True, to_file=verifiedFolder('runtime_' + self.params.runtime + '/modelArchitecture/' + self.name + '/gan.png')
        )

        if(not self.params.continuing):
            self.saveModel()

    #treinamento GAN
    def train(self, dataset: Dataset):
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
            print("starting epoch" + str(epoch))
            if(loadParam('close') == True):
                saveParam('close', False)
                self.saveModel(epoch-1, genLossHist, discLossHist)
                sys.exit()
            for i in range(nBatches):
                for j in range(self.extraDiscEpochs):
                    imgBatch, labelBatch = dataset.getTrainData((i+j)*self.batchSize, (i+j+1)*self.batchSize)

                    realImagesShuffled, realLabelsShuffled = shuffle_no_repeat(imgBatch, labelBatch)
                    trueInstances = [realImagesShuffled, realLabelsShuffled, imgBatch, labelBatch]
                    
                    genInput = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                    genImgOutput = self.generator.predict([genInput, labelBatch, imgBatch], verbose=0)
                    generatedInstances = [imgBatch, labelBatch, genImgOutput, labelBatch]
                    
                    XImg = np.concatenate((trueInstances[0],generatedInstances[0]))
                    XLabel = np.concatenate((trueInstances[1], generatedInstances[1]))
                    
                    XImg2 = np.concatenate((trueInstances[2],generatedInstances[2]))
                    XLabel2 = np.concatenate((trueInstances[3], generatedInstances[3]))

                    y = ([-1] * self.batchSize) + ([1] * self.batchSize)
                    y = np.reshape(y, (-1,))

                    (XImg, XLabel, XImg2, XLabel2, y) = shuffle(XImg, XLabel, XImg2, XLabel2, y)
                    discLoss = self.discriminator.train_on_batch([XImg,XLabel, XImg2, XLabel2], y)

                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                        l.set_weights(weights)
                
                genTrainNoise = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))

                gentrainLbls = [-1]*(self.batchSize)
                gentrainLbls = np.reshape(gentrainLbls, (-1,))
                ganLoss = self.gan.train_on_batch([genTrainNoise, labelBatch, imgBatch],gentrainLbls)

                if i == nBatches-1:
                    discLossHist.append(discLoss)
                    genLossHist.append(ganLoss)

                    print("Epoch " + str(epoch) + "\nCGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss))
                    infoFile = open(self.basePath + '/info.txt', 'a')
                    infoFile.write("Epoch " + str(epoch) + "\nCGAN (generator training) loss: " + str(ganLoss) + "\ndiscriminator loss: " + str(discLoss)+ '\n')
                    infoFile.close()

                    images = self.generator.predict([benchNoise, labelBatch, imgBatch])
                    out = ((images * 127.5) + 127.5).astype('uint8')
                    showOutputAsImg(out, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a) for a in labelBatch[:20]]) + '.png', colored=True)
                    
                    plotLoss([[genLossHist, 'generator loss'],[discLossHist, 'discriminator loss']], self.basePath + '/trainPlot.png')

            if((self.params.saveModels and epoch%5 == 0) or epoch == self.ganEpochs-1):
                self.saveModel(epoch, genLossHist, discLossHist)
                
    #Gera e salva imagens
    def saveGenerationExample(self, nEntries = 20):
        noise = np.random.uniform(-1,1, size=(5*self.nClasses,self.noiseDim))
        labels = np.floor(np.array(range(5*self.nClasses))/5)
        images = self.generator.predict([noise, labels])
        out = ((images * 127.5) + 127.5).astype('uint8')
        showOutputAsImg(out, self.basePath + '/finalOutput_f' + str(self.currentFold) + '_' + '_'.join([str(a) for a in labels]) + '.png',self.nClasses*5, colored=True)

    def generate(self, nEntries):
        print(self.name + ": started data generation")
        genInput = np.random.uniform(-1,1,size=(nEntries,self.noiseDim))
        genLabelInput = np.random.randint(0,self.nClasses, size = (nEntries))
        #genLabelInput = np.array([[1 if i == li else -1 for i in range(self.nClasses)] for li in genLabelInput], dtype='float32')

        if(self.generator is None):
            self.compile()
        genImages = self.generator.predict([genInput, genLabelInput])
        print(self.name + ": finished data generation")
        return genImages, genLabelInput