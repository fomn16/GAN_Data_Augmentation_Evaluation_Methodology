import sys
sys.path.insert(1, '../../')
from Modules.Shared.helper import *
import json

def saveParam(param, value):
    try:
        with open('savefile.json', 'r') as file:
            data = json.load(file)
    except:
        data = {}

    data[param] = value

    with open('savefile.json', 'w') as file:
        json.dump(data, file, indent=4)

def loadParam(param):
    try:
        with open('savefile.json', 'r') as file:
            data = json.load(file)
        return data[param]
    except:
        return None