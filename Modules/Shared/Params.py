import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *

class Params:
    nClasses = None
    kFold = None
    currentFold = None
    imgChannels = None
    imgWidth = None
    imgHeight = None
    dataDir = None
    datasetName = None
    runtime = None
    saveModels = None
    continuing = None