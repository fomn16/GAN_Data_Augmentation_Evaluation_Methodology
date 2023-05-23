import sys
import math
import cv2
import seaborn as sns
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.manifold import TSNE

class TSNE_INCEPTION(Benchmark):
    nClasses = 10
    enlargedWidth = 84
    enlargedHeight = 84
    #carrega dataset
    inceptionBatchSize = 250

    def __init__(self, params: Params, nameComplement = ""):
        self.name = self.__class__.__name__ + nameComplement

        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name + '/fold_' + str(params.currentFold))

        self.inceptionModel = InceptionV3(input_shape = (self.enlargedWidth, self.enlargedHeight, 3), include_top = False)
        self.params = params

    def train(self, augmentator: Augmentator, dataset: Dataset, extraEpochs = 1):
        p = None
        labels = None
        nEntries = int(np.floor(dataset.totalInstances/self.params.kFold))
        for i in range(math.floor(nEntries/self.inceptionBatchSize)):
            resizedImgs = []
            testImgs, testLbls = dataset.getTrainData(self.inceptionBatchSize*i, min(self.inceptionBatchSize*(i+1), nEntries))
            
            for img in testImgs:
                retyped = ((img * 127.5) + 127.5).astype('uint8')
                resized = cv2.resize(retyped, dsize=(self.enlargedWidth, self.enlargedHeight), interpolation=cv2.INTER_CUBIC)
                del retyped
                reshaped = np.expand_dims(resized, axis=-1)
                del resized
                untyped = (reshaped.astype('float') - 127.5)/127.5
                del reshaped
                resizedImgs.append(untyped)
                del untyped

            genImgs, genLbls = augmentator.generate(self.inceptionBatchSize)
            genLbls += self.nClasses

            for img in genImgs:
                retyped = ((img * 127.5) + 127.5).astype('uint8')
                resized = cv2.resize(retyped, dsize=(self.enlargedWidth, self.enlargedHeight), interpolation=cv2.INTER_CUBIC)
                del retyped
                reshaped = np.expand_dims(resized, axis=-1)
                del resized
                untyped = (reshaped.astype('float') - 127.5)/127.5
                del reshaped
                resizedImgs.append(untyped)
                del untyped
            #del genImgs

            resizedImgs = np.array(resizedImgs)
            if(self.params.imgChannels == 1):
                resizedImgs = np.concatenate((resizedImgs,)*3, axis=-1)
    
            out = self.inceptionModel.predict(resizedImgs)
            del resizedImgs

            out = np.reshape(out,(out.shape[0], out.shape[-1]))
            if(i == 0):
                p = out
                labels = testLbls
                labels =  np.concatenate((labels, genLbls))
            else:
                p = np.concatenate((p, out))
                labels =  np.concatenate((labels, testLbls))
                labels =  np.concatenate((labels, genLbls))
            del out
            #del genLbls
            print("Generating inception descriptors, batch " + str(i) + "/" + str(math.ceil(nEntries/self.inceptionBatchSize))+'\n')
            infoFile = open(self.basePath + '/info.txt', 'a')
            infoFile.write("Generating inception descriptors, batch " + str(i) + "/" + str(math.ceil(nEntries/self.inceptionBatchSize))+'\n')
            infoFile.close()
        del testImgs
        del testLbls

        tsne = TSNE(n_components=2, verbose=1, random_state=1602)
        z = tsne.fit_transform(p)
        del p

        plt.clf()
        df = pd.DataFrame()
        df["y"] = labels
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("colorblind", 2* self.nClasses),
                        data=df,s=5,alpha=0.3).set(title="Projeção T-SNE do dataset original e gerado")

        plt.savefig(verifiedFolder(self.basePath + '/' + augmentator.name + '/todos.png'))
        plt.clf()

        for j in range(self.nClasses):
            tstZ = []
            tstLbl = []
            for i in range(len(labels)):
                if(labels[i]%self.nClasses == j):
                    tstZ.append(z[i])
                    tstLbl.append(labels[i])
            tstZ = np.array(tstZ)
            tstLbl = np.array(tstLbl)

            df2 = pd.DataFrame()
            df2["y"] = tstLbl
            df2["comp-1"] = tstZ[:,0]
            df2["comp-2"] = tstZ[:,1]
            sns.scatterplot(x="comp-1", y="comp-2", hue=df2.y.tolist(),
                            palette=sns.color_palette("colorblind", 2),
                            data=df2,s=5,alpha=0.3).set(title="Projeção T-SNE do dataset original e gerado classe " + str(j))

            plt.savefig(verifiedFolder(self.basePath + '/' + augmentator.name + '/classe_' + str(j) + '.png'))

            plt.clf()