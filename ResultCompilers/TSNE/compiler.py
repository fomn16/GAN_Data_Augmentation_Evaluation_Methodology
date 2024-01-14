import os
from PIL import Image, ImageDraw, ImageFont
import sys

names = ['MNIST', 'CIFAR', 'PLANT', 'EUROSAT']
unbalanced = [True, False]

def nameSelector(name):
    condition = datasetName in name
    if datasetName != 'PLANT':
        condition = condition and not 'PLANT' in name  
    return condition

def getFilenames(condition):
    files = os.listdir(path)
    files = [f for f in files if condition(f)]
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

for datasetName in names:
    for b in unbalanced:
        if(b):
            path = '/home/fomn/TCC/TCC_Modulos/runtime_08_12_2023_15_48_56/trainingStats/TSNE_INCEPTION_(unbalanced)/fold_0'
        else:
            path = '/home/fomn/TCC/TCC_Modulos/runtime_08_12_2023_15_48_56/trainingStats/TSNE_INCEPTION_(default)/fold_0'

        files = orderNames(getFilenames(nameSelector))

        imgs = []
        width = None
        height = None
        mode = None
        f = True
        classes = 0
        for file in files:
            imgPath = os.listdir(path + '/' + file)
            imgPath = [f for f in imgPath if f.startswith('classe')]
            imgPath.sort()
            currImgs = []
            for img in imgPath:
                curImg = Image.open('/'.join([path, file, img]))
                if(f):
                    width, height = curImg.size
                    mode = curImg.mode
                    f = False
                currImgs.append(curImg)
            classes = len(currImgs)
            pasted = Image.new(mode, (width*classes, height))
            id = 0
            for img in currImgs:
                pasted.paste(img, (id*width,0))
                id+=1
            imgs.append(pasted)

        out = Image.new(mode, (imgs[0].size[0], imgs[0].size[1]*len(imgs)), 'white')
        id = 0
        for img in imgs:
            out.paste(img, (0, id*height))
            id+=1

        op = '_u' if b else ''
        out.save(os.getcwd()+'/output/'+datasetName.lower()+op+'.png')