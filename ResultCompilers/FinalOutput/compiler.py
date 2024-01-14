import os
from PIL import Image
import sys

path = '/home/fomn/TCC/TCC_Modulos/runtime_08_12_2023_15_48_56/trainingStats'
names = ['MNIST', 'CIFAR', 'PLANT', 'EUROSAT']
balanceOptions = [True, False]

def nameSelector(name):
    condition = datasetName in name
    if datasetName != 'PLANT':
        condition = condition and not 'PLANT' in name  
    if unbalanced:
        condition = condition and 'unbalanced' in name
    else:
        condition = condition and not 'unbalanced' in name
    return condition

def getFilenames(condition):
    files = os.listdir(path)
    files = [f for f in files if condition(f) and not(f.startswith('TSNE') or f.startswith('Classifier') or f.startswith('DATASET_DIRECTLY') or f.startswith('Discriminator'))]
    files = [f[::-1] for f in files]
    files.sort()
    files = [f[::-1] for f in files]
    return files

def orderNames(names):
    out = [None, None, None, None]
    for name in names:
        if(name.startswith('GAN')): out[0] = name
        if(name.startswith('CGAN')): out[1] = name
        if(name.startswith('WCGAN')): out[2] = name
        if(name.startswith('WUNET')): out[3] = name
    return out

def resize(img):
    if(datasetName == 'PLANT'):
        return img.crop((0,0,fWidth,oSize))
    
    reps = int(fWidth/(oSize*10))
    currReps = int(img.size[0]/(oSize*10))
    newImage = Image.new(oMode, (fWidth, oSize))
    id = 0
    for i in range(currReps*10):
        if((i%currReps) < reps):
            section = img.crop((i*oSize, 0, (i+1)*oSize, oSize))
            newImage.paste(section, (id*oSize,0))
            id+=1
    return newImage


for datasetName in names:
    for unbalanced in balanceOptions:
        files = orderNames(getFilenames(nameSelector))

        imgs = []
        for file in files:
            imgPath = os.listdir(path + '/' + file)
            imgPath = [f for f in imgPath if f.startswith('finalOutput')][0]
            imgs.append(Image.open('/'.join([path, file, imgPath])))

        oSize = imgs[0].size[1]
        oMode = imgs[0].mode
        fWidth = sys.maxsize
        for img in imgs:
            if(img.size[0] < fWidth):
                fWidth = img.size[0]

        outImage = Image.new(oMode, (fWidth, oSize*4))

        imgId = 0
        for img in imgs:
            if(img.size[0] == fWidth):
                outImage.paste(img, (0, imgId*oSize))
            else:
                outImage.paste(resize(img), (0, imgId*oSize))
            imgId+=1
        op = '_u' if unbalanced else ''
        outImage.save(os.getcwd()+'/output/'+datasetName.lower()+op+'.png')