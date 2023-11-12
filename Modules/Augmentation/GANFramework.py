import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Saving import *

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Shared.Params import Params

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

    for i in range(lbls.shape[0]):
        currClass = lbls[i]
        imgOutput[i] = sortedImgs[classLocations[currClass] + classCount[currClass]]
        classCount[currClass] += 1
    return imgOutput

def shuffle_different_class(imgs, lbls, classes):
    n = lbls.shape[0]
    indices = np.argsort(lbls)

    sortedImgs = imgs[indices]
    sortedLbls = lbls[indices]

    lstLbl = sortedLbls[0]
    lstLblId = 0
    classLocations = [0]*classes
    nClass = [0]*classes
    classLocations[sortedLbls[0]] = 0

    for i in range(n):
        nClass[sortedLbls[i]] += 1
        if(sortedLbls[i] != lstLbl):
            classLocations[sortedLbls[i]] = i
            lstLbl = sortedLbls[i]
            sortedImgs[lstLblId:i-1], sortedLbls[lstLblId:i-1] = shuffle_no_repeat(sortedImgs[lstLblId:i-1], sortedLbls[lstLblId:i-1])
            lstLblId = i
    sortedImgs[lstLblId:], sortedLbls[lstLblId:] = shuffle_no_repeat(sortedImgs[lstLblId:], sortedLbls[lstLblId:])

    imgOutput = np.ndarray(imgs.shape, imgs.dtype)
    lblOutput = np.ndarray(lbls.shape, lbls.dtype)
    for i in range(n):
        currClass = lbls[i]
        rand = 0
        if(n-nClass[currClass] > 0):
            rand = np.random.randint(n-nClass[currClass])
        indexPick = (rand + nClass[currClass] + classLocations[currClass])%n
        imgOutput[i] = sortedImgs[indexPick]
        lblOutput[i] = sortedLbls[indexPick]
    return imgOutput, lblOutput

class GANFramework(Augmentator):    
    def Block(self, model, nLayers: int, channels: int, kernelSize:int=3, kernelRegularizer=None):
        model = Conv2D(filters=channels, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides=2, kernel_regularizer=kernelRegularizer)(model)
        model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)
        for i in range(nLayers):
            model = Conv2D(filters=channels, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=kernelRegularizer)(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        return model
    
    def TransposedBlock(self, model, nLayers: int, channels: int, kernelSize:int=3, kernelRegularizer=None):
        model = Conv2DTranspose(filters=channels, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides=2, kernel_regularizer=kernelRegularizer)(model)
        model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        model = layers.Dropout(self.dropoutParam)(model)
        for i in range(nLayers):
            model = Conv2D(filters=channels, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=kernelRegularizer)(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        return model
    
    def ResidualBlock(self, model, nLayers:int, outDepth:int, kernelSize:int = 3, stride:int = 1):
        identity = model
        if(stride != 1):
            identity = layers.MaxPooling2D(stride)(identity)
        identity = Conv2D(filters=outDepth, kernel_size=1, padding='same', kernel_initializer='glorot_uniform')(identity)

        for i in range(nLayers):
            model = Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides = (stride if i == 0 else 1))(model)
            model = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(model)
            if(i != nLayers - 1):
                model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        
        model = layers.Dropout(self.dropoutParam)(model)
        model = layers.add([model, identity])
        model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        return model
    
    def InceptionBlock(self, model, nLayers:int, outDepth:int, stride:int = 1):
        for i in range(nLayers):
            identity = model
            if(stride != 1 and i == 0):
                identity = layers.MaxPooling2D(stride)(identity)
            identity = Conv2D(filters=outDepth, kernel_size=1, padding='same', kernel_initializer='glorot_uniform')(identity)
            paths = [identity]

            for kernelSize in [3, 5, 7]: 
                path = model
                path = Conv2D(filters=outDepth, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_uniform', strides = (stride if i == 0 else 1))(path)
                path = layers.BatchNormalization(axis=-1, epsilon=self.batchNormEpsilon, momentum=self.batchNormMomentum)(path)
                path = layers.Dropout(self.dropoutParam)(path)
                paths.append(path)

            model = layers.add(paths)
            model = layers.LeakyReLU(alpha=self.leakyReluAlpha)(model)
        return model
    
    def UNet(self, model, channels, channelRatio=2, nBlocks = 1):
        shape = tf.shape(model)._inferred_value
        spatialResolution = shape[-2]
        ksize = 3 if spatialResolution > 3 else spatialResolution
        downChannels = int(channels*channelRatio)

        model = self.ResidualBlock(model, nBlocks, channels, kernelSize=ksize)

        if(spatialResolution%2==0 and spatialResolution>=self.genWidth):
            down = layers.MaxPooling2D(2)(model)

            ret = self.UNet(down, downChannels, channelRatio)
            
            up = self.TransposedBlock(ret, 0, channels, ksize)

            model = layers.concatenate([model, up])

        model = self.ResidualBlock(model, nBlocks, channels, kernelSize=ksize)
        return model
    
    def saveModel(self, epoch = 0, genLossHist = [], discLossHist = []):
        saveParam(self.name + '_current_epoch', epoch)
        saveParam(self.name + '_gen_loss_hist', genLossHist)
        saveParam(self.name + '_disc_loss_hist', discLossHist)
        epochPath = self.basePath + '/modelSaves/fold_' + str(self.currentFold) + '/epoch_' + str(epoch)

        self.discriminator.save_weights(verifiedFolder(epochPath + '/disc_weights'))
        self.generator.save_weights(verifiedFolder(epochPath + '/gen_weights'))

        saveParam(self.name + '_disc_opt_lr', np.float64(self.optDiscr._decayed_lr('float32').numpy()))
        saveParam(self.name + '_gan_opt_lr', np.float64(self.optGan._decayed_lr('float32').numpy()))