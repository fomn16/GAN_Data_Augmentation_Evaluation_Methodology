import sys
import math
import cv2
import seaborn as sns
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

from keras.applications.inception_v3 import InceptionV3
from sklearn.manifold import TSNE
import trimap

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Benchmark.Benchmark import Benchmark
from Modules.Shared.Params import Params

class TSNE_INCEPTION(Benchmark):
    nClasses = 10
    enlargedWidth = 84
    enlargedHeight = 84
    #carrega dataset
    inceptionBatchSize = 250

    def __init__(self, params: Params, nameComplement = ""):
        self.name = self.__class__.__name__ + "_" +  nameComplement

        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name + '/fold_' + str(params.currentFold))

        self.inceptionModel = InceptionV3(input_shape = (self.enlargedWidth, self.enlargedHeight, 3), include_top = False)
        self.params = params

    def generateDescriptors(self, imgs):
        resizedImgs = []
        for img in imgs:
            retyped = ((img * 127.5) + 127.5).astype('uint8')
            resized = cv2.resize(retyped, dsize=(self.enlargedWidth, self.enlargedHeight), interpolation=cv2.INTER_CUBIC)
            del retyped
            reshaped = np.expand_dims(resized, axis=-1)
            del resized
            untyped = (reshaped.astype('float') - 127.5)/127.5
            del reshaped
            resizedImgs.append(untyped)
            del untyped
        resizedImgs = np.array(resizedImgs)

        if(self.params.imgChannels == 1):
            resizedImgs = np.concatenate((resizedImgs,)*3, axis=-1)
        out = self.inceptionModel.predict(resizedImgs)
        del resizedImgs

        return np.reshape(out,(out.shape[0], out.shape[-1]))

    def train(self, augmentator: Augmentator, dataset: Dataset, extraEpochs = 1):
        #0 = dataset, 1 = gerado
        p = [None, None]    #embeddings
        l = [None, None]    #labels

        nBatches = int(dataset.testInstances/self.inceptionBatchSize)
        for i in range(nBatches):
            testImgs, testLbls = dataset.getTestData(i*self.inceptionBatchSize, (i+1)*self.inceptionBatchSize)
            out = self.generateDescriptors(testImgs)
            if(i == 0):
                p[0] = out
                l[0] = testLbls
            else:
                p[0] = np.concatenate((p[0], out))
                l[0] =  np.concatenate((l[0], testLbls))

            genImgs, genLbls = augmentator.generate(self.inceptionBatchSize)
            genLbls += self.nClasses
            out = self.generateDescriptors(genImgs)
            if(i == 0):
                p[1] = out
                l[1] = genLbls
            else:
                p[1] = np.concatenate((p[1], out))
                l[1] = np.concatenate((l[1], genLbls))
           
            print("Generating inception descriptors, batch " + str(i) + "/" + str(nBatches)+'\n')
            infoFile = open(self.basePath + '/info.txt', 'a')
            infoFile.write("Generating inception descriptors, batch " + str(i) + "/" + str(nBatches)+'\n')
            infoFile.close()
        del testImgs
        del testLbls
        del genImgs
        del genLbls
        del out

        embeddings = np.concatenate((p[0], p[1]))
        labels = np.concatenate((l[0], l[1]))

        tsne = TSNE(n_components=2, verbose=1, random_state=1602)
        low_dim_embeddings = tsne.fit_transform(embeddings)

        #calcula e imprime benchmarks sobre embeddings
        
        #global score
        lde_dts = tsne.fit_transform(p[0])  #embeddings dataset
        lde_gen = tsne.fit_transform(p[1])  #embeddings dados gerados
        gs = trimap.TRIMAP(verbose=False).global_score(embeddings, low_dim_embeddings)
        gs_dts = trimap.TRIMAP(verbose=False).global_score(p[0], lde_dts)
        gs_den = trimap.TRIMAP(verbose=False).global_score(p[1], lde_gen)
        gs_comp = trimap.TRIMAP(verbose=False).global_score(p[0], lde_gen)
        message = "Trimap gloal score.\n-combined embeddings: " + str(gs) + "\n-dataset embeddings: " + str(gs_dts) + "\n-generated image embeddings: " + str(gs_den) + "\n-comparative embeddings: " + str(gs_comp)
        print(message)
        infoFile = open(self.basePath + '/info.txt', 'a')
        infoFile.write(message)
        infoFile.close()

        del p

        plt.clf()
        df = pd.DataFrame()
        df["y"] = labels
        df["comp-1"] = low_dim_embeddings[:,0]
        df["comp-2"] = low_dim_embeddings[:,1]
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
                    tstZ.append(low_dim_embeddings[i])
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