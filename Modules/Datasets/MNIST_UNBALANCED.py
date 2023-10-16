import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
from Modules.Shared.Params import Params
from Modules.Datasets.Dataset import Dataset


def AlterDataset(name, with_info,as_supervised,data_dir,nameComplement,transformationFunction):
    outputDir = f'{data_dir}/{name}_{nameComplement}_tfrecords'
    
    dataset, info = tfds.load(name=name, with_info=with_info, as_supervised=as_supervised, data_dir=data_dir)
    train = dataset['train']
    test = dataset['test']
    print(info)
    #se o dataset ainda não foi criado
    if not os.path.exists(outputDir):

        alteredTrain = train.map(transformationFunction)
        alteredTest = test.map(transformationFunction)

        filteredTrainList = list(alteredTrain.filter(lambda x, y: y != -1).as_numpy_iterator())
        filteredTest = alteredTest.filter(lambda x, y: y != -1)

    AlteredTrain = tf.data.experimental.tf(verifiedFolder(outputDir + '/train'))
    AlteredTest = tf.data.experimental.TFRecordDataset(verifiedFolder(outputDir + '/test'))

    return (AlteredTrain, AlteredTest), info

class MNIST_UNBALANCED(Dataset):
    def __init__(self, params:Params):
        self.params = params
        self.loadParams()

        self.name = params.datasetName

        def unbalance(image, label):
            if np.random.rand() < (label/10):  #coef = (nClasses - 1)/(1 - quantia restante da classe mais apagada)
                return image, label
            else:
                return image, tf.constant(-1, dtype=tf.int64)
             
        self.dataset, self.info = AlterDataset(params.datasetName, True, True, params.dataDir, 'unbalanced', unbalance)

        #numero de instancias nos splits de treinamento e teste no dataset original
        self.trainInstancesDataset = self.info.splits['train'].num_examples
        self.testInstancesDataset = self.info.splits['test'].num_examples
        #número total de instâncias
        self.totalInstances = int(np.floor((self.trainInstancesDataset + self.testInstancesDataset)))
        #número de instâncias em cada divisão do fold que vai para treinamento
        self.n_instances_fold_train = int(np.floor(self.totalInstances/self.params.kFold))
        #numero de instâncias de treinamento nesse fold
        self.trainInstances = self.n_instances_fold_train*(self.params.kFold - 1)
        #numero de instâncias de teste nesse fold
        self.testInstances = self.totalInstances - self.trainInstances

        self.trainDataset = getFromDatasetLL(0, self.trainInstances - 1, params.currentFold, self.n_instances_fold_train,
                                             self.testInstances, self.trainInstancesDataset, params.nClasses,
                                             self.dataset, 0, 1)
        
        self.testDataset = getFromDatasetLL(0, self.testInstances - 1, params.currentFold, self.n_instances_fold_train,
                                             self.testInstances, self.trainInstancesDataset, params.nClasses,
                                             self.dataset, 0, 1, test=True)
    
    def loadParams(self):
        self.params.datasetName = 'mnist'
        self.params.nClasses = 10
        self.params.imgChannels = 1
        self.params.imgWidth = 28
        self.params.imgHeight = 28
    
    def getTrainData(self, start, end):
        imgs, lbls = self.trainDataset
        return imgs[start:end], lbls[start:end]

    def getTestData(self, start, end):
        imgs, lbls = self.testDataset
        return imgs[start:end], lbls[start:end]