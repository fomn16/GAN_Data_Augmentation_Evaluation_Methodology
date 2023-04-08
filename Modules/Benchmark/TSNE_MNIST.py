import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.manifold import TSNE

class TSNE_MNIST:
    noiseDim = 100
    nClasses = 10
    imgWidth = 28
    imgHeight = 28

    enlargedWidth = 84
    enlargedHeight = 84
    #carrega dataset
    inceptionBatchSize = 250

    def _loadIntoArray(dataset):
        npI = dataset.as_numpy_iterator()
        imgs = []
        lbls = []
        for d in npI:
            imgs.append(d[0])
            lbls.append(d[1])
        imgs = np.array(imgs)
        imgs = (imgs.astype("float") - 127.5) / 127.5
        lbls = np.array(lbls)
        return imgs, lbls

    def __init__(self, dataset, info, params: Params, nameComplement = ""):
        self.name = self.__class__.__name__ + nameComplement

        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime/trainingStats/' + self.name)
        self.currentFold = params.currentFold

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight
        

        self.inceptionModel = InceptionV3(input_shape = (self.enlargedWidth, self.enlargedHeight, 3), include_top = False)

        #organizando imagens e labels para treinamento do tsne
        imgs1, lbls1 = self._loadIntoArray(dataset['train'])
        imgs2, lbls2 = self._loadIntoArray(dataset['test'])

        imgs = np.concatenate((imgs1, imgs2))
        lbls = np.concatenate((lbls1, lbls2))

        totalEntries = imgs.shape[0]
        n = int(np.floor(totalEntries/5))

        self.testImgs = imgs[:n]
        self.testLbls = lbls[:n]
        self.nEntries = self.testImgs.shape[0]

    def train(self, imgs, lbls, trainName, extraEpochs = 1):
        pass

    def runTest(self, imgs, lbls):
        pass