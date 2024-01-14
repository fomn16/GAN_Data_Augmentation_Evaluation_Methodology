import os
files = os.listdir(os.getcwd())
for file in files:
    if file.endswith('.txt'):
        with open(file, 'r') as f:
            fileData = []
            text = ''
            name = ''
            fl = True
            for line in f:
                if(fl):
                    name = line.replace('Fold 0: Starting Classifier training on WUNETCGAN_','').replace('Epoch 0', '')
                    print(name)
                    fl = False
                text += line
                if(line.startswith('auroc score: ')):
                    fileData.append((text, float(line.replace('auroc score: ',''))))
                    text = ''
            fileData = sorted(fileData, key = lambda x: x[1], reverse=True)
            with open('output/' + name + '.txt', 'w') as outFile:
                for i in fileData:
                    outFile.write(i[0])