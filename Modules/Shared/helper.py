#imports de todas as bibliotecas utilizadas:
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np
import IPython.display
import PIL.Image
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input
import matplotlib.pyplot as plt
import random
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
import os
import itertools

import sys
sys.path.insert(1, '../../')
from Modules.Shared.DatasetIteratorInfo import DatasetIteratorInfo


#Funções helper para apresentação de imagens e dados de treinamento

def concatArray(a, n):
    d = []
    for j in range(a.shape[1]):
        for i in range(n):
            d.append(a[i][j])
    d = np.array(d)
    d = np.reshape(d, (a.shape[1],a.shape[2]*n))
    return d

#mostrar input/output
def showOutputAsImg(out, path, n = 20):
  PIL.Image.fromarray(concatArray(out,n)).resize(size=(100*n,100)).save(path)

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
def loadIntoArrayLL(datasetSection, dataset, nClasses, start, end, iterators:DatasetIteratorInfo):
    if(datasetSection == 'test'):
        if(start >= iterators.lastTestId):
            iterators.testIter = itertools.islice(iterators.testIter, start - iterators.lastTestId, None)
        else:
            iterators.testIter = itertools.islice(dataset['test'].as_numpy_iterator(), start, None)
        iterators.lastTestId = end + 1
        imgs = []
        lbls = []
        for i in range(end - start):
            d = next(iterators.testIter)
            imgs.append(d['image'])
            lbls.append([int(d['label'] == n) for n in range(nClasses)])
        imgs = np.array(imgs).astype("float")
        imgs = (imgs - 127.5) / 127.5
        lbls = np.array(lbls)
        return imgs, lbls
    else:
        if(start >= iterators.lastTrainId):
            iterators.trainIter = itertools.islice(iterators.trainIter, start - iterators.lastTrainId, None)
        else:
            iterators.trainIter = itertools.islice(dataset['train'].as_numpy_iterator(), start, None)
        iterators.lastTrainId = end
        #rsdz = map(resizeImg, npI)
        imgs = []
        lbls = []
        for i in range(end - start):
            d = next(iterators.trainIter)
            imgs.append(d['image'])
            lbls.append([int(d['label'] == n) for n in range(nClasses)])
        imgs = np.array(imgs).astype("float")
        imgs = (imgs - 127.5) / 127.5
        lbls = np.array(lbls)
        return imgs, lbls