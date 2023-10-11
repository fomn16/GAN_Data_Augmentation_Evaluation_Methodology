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

def shuffle_same_class(imgs, lbls, classes):
    indices = np.argsort(lbls)

    sortedImgs = imgs[indices]
    sortedLbls = lbls[indices]

    lstLbl = sortedLbls[0]
    lstLblId = 0
    classLocations = [0]*classes
    classLocations[sortedLbls[0]] = 0

    for i in range(sortedLbls.shape[0]):
        if(sortedLbls[i] != lstLbl):
            classLocations[sortedLbls[i]] = i
            lstLbl = sortedLbls[i]
            sortedImgs[lstLblId:i-1], sortedLbls[lstLblId:i-1] = shuffle_no_repeat(sortedImgs[lstLblId:i-1], sortedLbls[lstLblId:i-1])
            lstLblId = i
    sortedImgs[lstLblId:], sortedLbls[lstLblId:] = shuffle_no_repeat(sortedImgs[lstLblId:], sortedLbls[lstLblId:])

    classCount = [0]*classes
    imgOutput = np.ndarray(imgs.shape, imgs.dtype)
    lblOutput = np.copy(lbls)

    for i in range(lbls.shape[0]):
        currClass = lbls[i]
        imgOutput[i] = sortedImgs[classLocations[currClass] + classCount[currClass]]
        classCount[currClass] += 1
    return imgOutput, lblOutput

class WUNETCGAN_MNIST(Augmentator):
    #Constantes:
    genWidth = 7
    genHeight = 7
    embeddingDims = 32

    approximateNoiseDim = 100
    noiseDepth = int(np.ceil(approximateNoiseDim/(genWidth*genHeight)))
    noiseDim = genWidth*genHeight*noiseDepth

    initLr = 2.5e-5
    leakyReluAlpha = 0.2
    dropoutParam = 0.02
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
    
    def AddUNet(self, model, channels, channelRatio=2):
        shape = tf.shape(model)._inferred_value
        spatialResolution = shape[-2]
        #inputChannels = shape[-1]
        ksize = 3 if spatialResolution > 3 else spatialResolution
        downChannels = int(channels*channelRatio)

        identity = model
        #if(inputChannels != channels):
        identity = Conv2D(filters=channels, kernel_size=1, padding='same', kernel_initializer='glorot_uniform')(identity)

        model = Conv2D(filters=channels, kernel_size=ksize, padding='same', kernel_initializer='glorot_uniform')(model)
        model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)

        if(spatialResolution%2==0):
            down = Conv2D(filters=downChannels, kernel_size=ksize, padding='same', kernel_initializer='glorot_uniform', strides=2)(model)
            down = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(down)
            down = layers.LeakyReLU(alpha=self.leakyReluAlpha)(down)

            ret = self.AddUNet(down, downChannels, channelRatio)
            
            up = Conv2DTranspose(filters=channels, kernel_size=ksize, padding='same', kernel_initializer='glorot_uniform', strides=2)(ret)
            up = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(up)
            up = layers.LeakyReLU(alpha=self.leakyReluAlpha)(up)

            model = layers.concatenate([model, up])

        model = Conv2D(filters=channels, kernel_size=ksize, padding='same', kernel_initializer='glorot_uniform')(model)
        model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
        model = layers.add([model, identity])
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        return model

    def AddResidualBlock(self, model, nLayers:int, outDepth:int, kernelSize:int = 3, stride:int = 1):
        identity = model
        if(stride != 1):
            identity = layers.AveragePooling2D(stride)(identity)
        identity = Conv2D(filters=outDepth, kernel_size=1, padding='same', kernel_initializer='glorot_uniform')(identity)

        for i in range(nLayers):
            model = Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides = (stride if i == 0 else 1))(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            if(i != nLayers - 1):
                model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)

        model = layers.add([model, identity])
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)
        return model
    
    def AddTransposedBlock(self, model, nLayers: int, channels: int, kernelSize:int=3):
        model = Conv2DTranspose(filters=channels, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides=2)(model)
        model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)
        for i in range(nLayers):
            model = Conv2D(filters=channels, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform')(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        return model
    
    #Cria model geradora com keras functional API
    def createGenModel(self):
        noise = keras.Input(shape=(self.noiseDim,), name = 'gen_input_gaussian_distribution')
        # Faz reshape para dimensões espaciais desejadas
        X = layers.Reshape((self.genWidth, self.genHeight, self.noiseDepth))(noise)
    
        label = keras.Input(shape=(1,), name = 'gen_input_label')
        embeddedLabels= layers.Embedding(self.nClasses, self.genWidth*self.genHeight)(label)
        reshapedLabels = layers.Reshape((self.genWidth, self.genHeight, 1))(embeddedLabels)
        X = layers.concatenate([X, reshapedLabels])

        X = self.AddTransposedBlock(X, 1, 8)
        X = self.AddTransposedBlock(X, 1, 4)

        imageInput = keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgChannels), name = 'gen_input_img')

        X = layers.concatenate([X, imageInput])

        X = self.AddUNet(X, 32, 1.5)

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

        label = keras.Input(shape=(1,), name = 'disc_input_label')
        embeddedLabel = layers.Embedding(self.nClasses, self.imgWidth*self.imgHeight)(label)
        reshapedLabel = layers.Reshape((self.imgWidth, self.imgHeight, 1))(embeddedLabel)

        X = layers.concatenate([img1, img2, reshapedLabel])

        X = self.AddResidualBlock(X, 1, 64, stride=2)
        X = self.AddResidualBlock(X, 1, 64, stride=2)

        X = Conv2D(1, 7, kernel_initializer='glorot_uniform', activation='linear')(X)
        discOutput = Flatten(name = 'discoutput_realvsfake')(X)

        self.discriminator = keras.Model(inputs = [img1, img2, label], outputs = discOutput, name = 'discriminator')
        self.discriminator.summary()
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
        cganOutput =  self.discriminator([self.generator([cganNoiseInput, cganLabelInput, cganImgInput]), cganImgInput, cganLabelInput])
        self.gan = Model((cganNoiseInput, cganLabelInput, cganImgInput), cganOutput)

        self.gan.compile(loss=wasserstein_loss, optimizer=self.optcGan)
        
        self.discriminator.trainable = True
        self.gan.summary()
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

                    realImagesShuffled, realLabelsShuffled = shuffle_same_class(imgBatch, labelBatch, self.nClasses)
                    
                    genInput = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                    fakeImgs = self.generator.predict([genInput, labelBatch, imgBatch], verbose=0)
                    
                    XImg = np.concatenate((realImagesShuffled, fakeImgs))
                    XImg2 = np.concatenate((imgBatch, imgBatch))
                    XLabel = np.concatenate((labelBatch, labelBatch))

                    y = ([-1] * self.batchSize) + ([1] * self.batchSize)
                    y = np.reshape(y, (-1,))

                    (XImg, XImg2, XLabel, y) = shuffle(XImg, XImg2, XLabel, y)
                    discLoss = self.discriminator.train_on_batch([XImg, XImg2, XLabel], y)

                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clipValue, self.clipValue) for w in weights]
                        l.set_weights(weights)
                
                imgBatch, labelBatch = dataset.getTrainData((i)*self.batchSize, (i+1)*self.batchSize)
                genTrainNoise = np.random.uniform(-1,1,size=(self.batchSize,self.noiseDim))
                y = [-1]*(self.batchSize)
                y = np.reshape(y, (-1,))
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
                    newOut = np.ndarray(out.shape, out.dtype)
                    for outId in range(10):
                        newOut[outId*2] = imgBatch[outId]
                        newOut[outId*2+1] = out[outId]
                    showOutputAsImg(newOut, self.basePath + '/output_f' + str(self.currentFold) + '_e' + str(epoch) + '_' + '_'.join([str(a) for a in labelBatch[:10]]) + '.png')
                    
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