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
from typing import List

import sys
sys.path.insert(1, '../../')

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
def showOutputAsImg(out, path, n = 20, colored = False):
  PIL.Image.fromarray(concatArray(out,n, colored)).resize(size=(100*n,100)).save(path)

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
def loadIntoArrayLL(datasetSection, dataset, nClasses, start, end, imgId, lblId):
     # create empty arrays for images and labels
    output_shapes = next(dataset[datasetSection].as_numpy_iterator())[imgId].shape
    imgs = np.zeros((end - start,) + output_shapes).astype('float')
    lbls = np.zeros((end - start, nClasses)).astype('int')
    
    # load data into arrays
    iterator = dataset[datasetSection].as_numpy_iterator()
    for i, instance in enumerate(itertools.islice(iterator, start, end)):
        img = instance[imgId]
        label = instance[lblId]
        img = np.expand_dims(img.astype('float'), axis=0)
        img = (img - 127.5) / 127.5
        imgs[i] = img
        lbls[i, label] = 1
    return imgs, lbls