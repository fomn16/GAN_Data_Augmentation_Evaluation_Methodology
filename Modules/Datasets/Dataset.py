import sys
sys.path.insert(1, '../../')
from Modules.Shared.Params import Params

class Dataset:
    trainInstances = 0
    testInstances = 0
    def __init__(self, params:Params):
        pass
    
    def getTrainData(self, start, end):
        pass

    def getTestData(self, start, end):
        pass
    
    def loadParams(self):
        pass