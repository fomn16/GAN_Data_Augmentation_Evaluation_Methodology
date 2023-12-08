#imports de todas as bibliotecas utilizadas:
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Flatten, Dropout, Embedding
from keras.optimizers import Adam, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np
import IPython.display
import PIL.Image
from tensorflow import keras
from keras import layers
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
from keras import backend as K
import pickle

'''
sudo apt install gcc
sudo apt install g++
pip install tensorflow tensorflow-addons scikit-learn pandas ipython pillow matplotlib tensorflow_datasets imgaug seaborn pydot annoy trimap
'''

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

def save_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def sliceForEntry(id, nEntryArray):
    for i in range(len(nEntryArray)):
        id -= nEntryArray[i]
        if(id <= 0):
            return i
    return len(nEntryArray) -1

def LoadDataset(name, with_info,as_supervised,data_dir,nameComplement,sliceNames, transformationFunction = None, filterFunction = None):
    outputDir = f'{data_dir}/{name}_{nameComplement}_tfrecords'
    
    dataset, info = tfds.load(
        name=name, 
        with_info=with_info, 
        as_supervised=as_supervised, 
        data_dir=data_dir)
    
    nEntries = 0
    nEntryArray = []
    loadedSlices = []
    
    for slice in sliceNames:
        if(slice in dataset):
            loadedSlices.append(dataset[slice])
            nEntryArray.append(info.splits[slice].num_examples)
            nEntries += info.splits[slice].num_examples

    maxStorageGigs = 5
    maxStorage = 125000000*maxStorageGigs

    retData = None

    #se o dataset ainda não foi criado
    if not os.path.exists(outputDir):
        alteredSlices = []
        for slice in loadedSlices:
            if(transformationFunction != None):
                alteredSlices.append(map(transformationFunction, slice.as_numpy_iterator()))
            else:
                alteredSlices.append(slice.as_numpy_iterator())

        nPixels = 1
        alteredSlices[0], tst = itertools.tee(alteredSlices[0])
        shape = next(tst)[0].shape
        for n in shape:
            nPixels *= n
        maxStorageImgs = int(np.floor(maxStorage/nPixels))

        includedEntries = 0
        entries = []
        while(includedEntries < maxStorageImgs):
            targetSlice = sliceForEntry(includedEntries, nEntryArray)
            if(includedEntries + nEntryArray[targetSlice] <= maxStorageImgs):
                includedEntries += nEntryArray[targetSlice]
                entries.extend(list(alteredSlices[targetSlice]))
            else:
                toInclude = maxStorageImgs - includedEntries
                includedEntries = maxStorageImgs
                entries.extend(list(itertools.islice(alteredSlices[targetSlice], toInclude)))

        imgs = np.array([i[0] for i in entries])
        lbls = np.array([i[1] for i in entries])

        del entries, alteredSlices, loadedSlices, dataset, info
        if(filterFunction != None):
            imgs, lbls = filterFunction(imgs, lbls)
            
        imgs = imgs.astype('float')
        imgs = (imgs - 127.5) / 127.5
        retData = imgs, lbls
        dataset = tf.data.Dataset.from_tensor_slices(retData)
        
        dataset.save(verifiedFolder(outputDir))
    else:
        dataset = tf.data.Dataset.load(verifiedFolder(outputDir))
        imgs = []
        lbls = []
        for i in dataset:
            imgs.append(i[0])
            lbls.append(i[1])
        imgs, lbls = np.array(imgs), np.array(lbls)
        retData = imgs, lbls
    return retData

def unbalance(imgs, lbls, minClassPercent, nClasses):
    tempCounter = [0]*nClasses
    for i in range(len(lbls)):
        tempCounter[lbls[i]] += 1
    maxClassCount = np.max(tempCounter)
    minClassCount = np.min(tempCounter)
    minClassInstances = int(minClassCount*minClassPercent)
    coeff = (maxClassCount - minClassInstances)/(nClasses-1)
    unbalancedImgs = []
    unbalancedLbls = []
    counter = [0]*nClasses
    for i in range(len(lbls)):
        id = lbls[i]
        if(counter[id] <= (id*coeff + minClassInstances)):
            unbalancedImgs.append(imgs[i])
            unbalancedLbls.append(lbls[i])
            counter[id] += 1
    unbalancedImgs = np.array(unbalancedImgs)
    unbalancedLbls = np.array(unbalancedLbls)
    return unbalancedImgs, unbalancedLbls

def resizeImg(imgSide, index, entry):
        img = entry[index]
        w = img.shape[0]
        h = img.shape[1]
        side = np.min([w,h])
        cw, ch = int(w/2 - side/2), int(h/2 - side/2)
        img = np.asarray(PIL.Image.fromarray(img[cw:cw+side, ch:ch+side]).resize(size=(imgSide,imgSide)))
        ret = None
        try:
            ret = entry[:index] + (img,) + entry[index + 1:]
        except:
            ret = entry
            ret[index] = img
        return ret

def addToName(s, pre = "_"):
    if(s is not None and s != ""):
        return pre + s
    return ""