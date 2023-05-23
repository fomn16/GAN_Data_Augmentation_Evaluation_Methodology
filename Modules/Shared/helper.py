#imports de todas as bibliotecas utilizadas:
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Flatten, Dropout, Embedding
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np
import IPython.display
import PIL.Image
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input
from keras import regularizers
import matplotlib.pyplot as plt
import random
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
import os
import itertools
from typing import List
from datetime import datetime
import tensorflow as tf
import sys
sys.path.insert(1, '../../')

from Modules.Datasets.Dataset import Dataset
from Modules.Augmentation.Augmentator import Augmentator
from Modules.Benchmark.Benchmark import Benchmark
from Modules.Shared.Params import Params

#Funções helper para apresentação de imagens e dados de treinamento
def concatArray(a, n, colored):
    d = []
    for j in range(a.shape[1]):
        for i in range(n):
            d.append(a[i][j])
    d = np.array(d)
    if (colored):
         d = np.reshape(d, (a.shape[1],a.shape[2]*n,a.shape[3]))
    else:
        d = np.reshape(d, (a.shape[1],a.shape[2]*n))
       
    return d

#mostrar input/output
def showOutputAsImg(out, path, n = 20, colored = False, mult = 10):
  w = out.shape[1]*mult
  h = out.shape[2]*mult
  PIL.Image.fromarray(concatArray(out,n, colored)).resize(size=(w*n,h)).save(path)

def plotLoss(losses, path, clear=True):
    if(clear):
        plt.clf()
    for loss in losses:
        plt.plot(loss[0], label=loss[1])
    plt.legend()
    plt.savefig(path)

def verifiedFolder(folderPath):
    createdFolder = folderPath
    if('.' in createdFolder.split('/')[-1]):
        createdFolder = '/'.join(createdFolder.split('/')[:-1])
    if not os.path.exists(createdFolder):
        os.makedirs(createdFolder)
    return folderPath

def loadIntoArray(dataset, nClasses):
    npI = dataset.as_numpy_iterator()
    imgs = []
    lbls = []
    for d in npI:
        imgs.append(d[0])
        lbls.append([int(d[1] == n) for n in range(nClasses)])
    imgs = np.array(imgs)
    imgs = (imgs.astype("float") - 127.5) / 127.5
    lbls = np.array(lbls)
    return imgs, lbls

#fazendo uso de lazy loading do dataset
def loadIntoArrayLL(datasetSection, dataset, nClasses, start, end, imgId, lblId, mapFunction = None):
    # create empty arrays for images and labels
    output_shapes = None
    iterator = None

    if(mapFunction == None):
        output_shapes = next(dataset[datasetSection].as_numpy_iterator())[imgId].shape
        iterator = dataset[datasetSection].as_numpy_iterator()
    else:
        output_shapes = next(map(mapFunction, dataset[datasetSection].as_numpy_iterator()))[imgId].shape
        iterator = map(mapFunction, dataset[datasetSection].as_numpy_iterator())
        
    imgs = np.zeros((end - start,) + output_shapes).astype('float')
    lbls = np.full((end - start,), 0).astype('int')
    
    for i, instance in enumerate(itertools.islice(iterator, start, end)):
        img = instance[imgId]
        label = instance[lblId]
        img = np.expand_dims(img.astype('float'), axis=0)
        img = (img - 127.5) / 127.5
        imgs[i] = img
        lbls[i] = label
    return imgs, lbls

#recupera o bloco do dataset considerando que os splits de treinamento e teste são juntos
def getBlockFromDataset(start, end, trainInstancesDataset, nClasses, dataset, mapFunction, imgId, lblId):
    imgs = None
    lbls = None
    #se todo o bloco a ser recuperado se encontra no split de treinamento do dataset
    if(end < trainInstancesDataset):
        imgs, lbls = loadIntoArrayLL('train', dataset, nClasses, start, end, imgId, lblId, mapFunction)
    #se todo o bloco a ser recuperado se encontra no split de teste do dataset    
    elif (start >= trainInstancesDataset):
        imgs, lbls = loadIntoArrayLL('test', dataset, nClasses, start - trainInstancesDataset, end - trainInstancesDataset, imgId, lblId, mapFunction)
    #se uma parte deve vir do split train e outra parte do test
    else:
        imgs1, lbls1 = loadIntoArrayLL('train', dataset, nClasses, start, trainInstancesDataset - 1, imgId, lblId, mapFunction)
        imgs2, lbls2 = loadIntoArrayLL('test', dataset, nClasses, 0, end - trainInstancesDataset + 1, imgId, lblId, mapFunction)
        imgs = np.concatenate((imgs1, imgs2))
        lbls = np.concatenate((lbls1, lbls2))
        del imgs1, imgs2, lbls1, lbls2
    return imgs, lbls

#carrega instancias do dataset usando lazy loading dos dados
def getFromDatasetLL(start, end, currentFold, n_instances_fold_train, testInstances, trainInstancesDataset, nClasses, dataset, imgId, lblId, mapFunction = None, test=False):
    imgs = None
    lbls = None
    testStart = currentFold*n_instances_fold_train
    testEnd = testStart + testInstances
    #para split de teste, desloca os índices pedidos para a área de inicio do split pedido, de acordo com o fold
    if(test):
        imgs, lbls = getBlockFromDataset(start + testStart, end + testStart, trainInstancesDataset, nClasses, dataset, mapFunction, imgId, lblId)
    #para split de treinamento
    else:
        #se a porção de treinamento consultada só requer dados antes da parte de testes do dataset
        if(end < testStart):
            imgs, lbls = getBlockFromDataset(start, end, trainInstancesDataset, nClasses, dataset, mapFunction, imgId, lblId)
        #se a porção de treinamento consultada só requer dados depois da parte de testes do dataset
        elif(start >= testStart):
            imgs, lbls = getBlockFromDataset(start + testInstances, end + testInstances, trainInstancesDataset, nClasses, dataset, mapFunction, imgId, lblId)
        #se a porção de treinamento consultada precisa de dados de antes e depois da parte de testes
        else:
            imgs, lbls = getBlockFromDataset(start, testStart - 1, trainInstancesDataset, nClasses, dataset, mapFunction, imgId, lblId)
            imgs2, lbls2 = getBlockFromDataset(testEnd, end + testInstances, trainInstancesDataset, nClasses, dataset, mapFunction, imgId, lblId)
            imgs = np.concatenate((imgs, imgs2))
            lbls = np.concatenate((lbls, lbls2))
            del imgs2, lbls2
    return imgs, lbls