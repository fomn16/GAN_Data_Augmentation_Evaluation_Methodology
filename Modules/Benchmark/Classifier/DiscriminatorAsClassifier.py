import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Benchmark.Benchmark import Benchmark

class DiscriminatorAsClassifier(Benchmark):


    def __init__(self, params: Params, nameComplement = ""):
        self.name = self.__class__.__name__ + addToName("(" +  nameComplement + ")")

        self.nClasses = params.nClasses
        self.basePath = verifiedFolder('runtime_' + params.runtime + '/trainingStats/' + self.name)
        self.currentFold = params.currentFold

        self.imgChannels = params.imgChannels
        self.imgWidth = params.imgWidth
        self.imgHeight = params.imgHeight

        self.params = params
        self.augmentator = None

    def train(self, augmentator: Augmentator, dataset: Dataset):
        if("WUNETCGAN" in augmentator.name and "MIXED" not in augmentator.name):
            self.augmentator = augmentator

    def runTest(self, dataset: Dataset):
        if(self.augmentator is not None):
            imgs, lbls = dataset.getTestData(0, dataset.testInstances)
            classOutput, _ = self.augmentator.discriminator.predict([imgs, imgs], verbose=0)
            lbls = np.array([[1 if i == lbl else 0 for i in range(self.nClasses)] for lbl in lbls], dtype='float32')

            stats = ""
            for i in range(20):
                stats += "["
                for j in classOutput[i]:
                    stats += str(j) + ", "
                
                stats += "] -> " + str(_[i]) + "\n"

            aurocScore = ""
            try:
                aurocScore = str(roc_auc_score(lbls, classOutput))
            except:
                aurocScore = "error calculating:\n" + str(lbls) + "\n" + str(classOutput)
            
            lbls = [[int(np.argmax(o) == i) for i in range(self.nClasses)] for o in lbls]
            classOutput = [[int(np.argmax(o) == i) for i in range(self.nClasses)] for o in classOutput]

            report = classification_report(lbls, classOutput) + '\nauroc score: ' + aurocScore  + "\ndisc output: " + stats+ '\n'

            print(report)
            infoFile = open(self.basePath + '/info.txt', 'a')
            infoFile.write(report)
            infoFile.close()