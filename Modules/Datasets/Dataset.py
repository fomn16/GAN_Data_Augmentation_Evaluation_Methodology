import sys
sys.path.insert(1, '../..')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Shared.config import *

class Dataset:
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()
        self.name = params.datasetName + addToName(self.params.datasetNameComplement)
    
    def loadParams(self):
        self.params.datasetName = Datasets.MNIST
        self.params.datasetNameComplement = 'default'

        self.params.nClasses = 10
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28

        self.transformFunction = None
        self.filterFunction = None

        self.slices = ['train', 'test']
    
    def getTrainData(self, start, end):
        return self.trainImgs[start:end].copy(), self.trainLbls[start:end].copy()

    def getTestData(self, start, end):
        return self.testImgs[start:end].copy(), self.testLbls[start:end].copy()
    
    def load(self):
        self.loadParams()
        imgs,lbls = LoadDataset(
                        self.params.datasetName, 
                        True, 
                        True, 
                        self.params.dataDir, 
                        self.params.datasetNameComplement, 
                        self.slices,
                        self.transformFunction,
                        self.filterFunction
                    )

        #número total de instâncias
        self.totalInstances = lbls.shape[0]
        #número de instâncias em cada divisão do fold que vai para treinamento
        self.n_instances_fold_train = int(np.floor((self.totalInstances/self.params.kFold)))
        #numero de instâncias de treinamento nesse fold
        self.trainInstances = self.n_instances_fold_train*(self.params.kFold - 1)
        #numero de instâncias de teste nesse fold
        self.testInstances = self.totalInstances - self.trainInstances

        testStart = self.params.currentFold*self.n_instances_fold_train
        testEnd = testStart + self.testInstances
        self.testImgs = imgs[testStart:testEnd]
        self.testLbls = lbls[testStart:testEnd]
        self.trainImgs = np.concatenate((imgs[:testStart],imgs[testEnd:]))
        self.trainLbls = np.concatenate((lbls[:testStart],lbls[testEnd:]))
        self.params.datasetTrainInstances = self.trainInstances

        #salvando estatísticas do dataset
        print(self.name + '\n')
        print('train: ' + str(self.trainInstances)+ '\n')
        print('test: ' + str(self.testInstances)+ '\n')
        print('total: ' + str(self.totalInstances)+ '\n')
        tst = [0]*self.params.nClasses
        for c in self.trainLbls:
            tst[c]+=1
        for c in self.testLbls:
            tst[c]+=1
        print('\n\n')

        if('unbalanced' not in self.name):
            plt.bar(range(self.params.nClasses), tst, label='original')
            plt.xlabel('Id da Classe')
            plt.ylabel('Número de Instâncias')
            plt.title(self.name.replace('_default', '').replace('_unbalanced', ' desbalanceado').replace('cifar10', 'CIFAR-10') + ' Instâncias x Classe')
        else:
            plt.bar(range(self.params.nClasses), tst, label='desbalanceado')

        if(self.params.nClasses>20):
            plt.xticks(range(self.params.nClasses),fontsize=6)
        else:
            plt.xticks(range(self.params.nClasses))

        if('unbalanced' in self.name):
            plt.legend(loc='upper left')
            plt.savefig(verifiedFolder('runtime_' + self.params.runtime + '/' + self.name + '_classGraph.png'))
            plt.clf()

    def unload(self):
        del self.testImgs, self.testLbls, self.trainImgs, self.trainLbls