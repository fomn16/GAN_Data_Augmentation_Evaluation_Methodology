import statistics
import os

path = '/home/fomn/TCC/TCC_Modulos/runtime_08_12_2023_15_48_56/trainingStats/TSNE_INCEPTION_(unbalanced)/fold_0/info.txt'

values = []

with open(path, 'r') as f:
    for line in f:
        if line.startswith('-combined embeddings:'):
            values.append(float(line.replace('-combined embeddings: ','')))

print(statistics.mean(values), statistics.stdev(values))