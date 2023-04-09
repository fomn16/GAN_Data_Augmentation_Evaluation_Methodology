class DatasetIteratorInfo:
    trainIter = None
    lastTrainId = None
    testIter = None
    lastTestId = None

    def __init__(self, tranIter, testIter):
        self.trainIter = tranIter
        self.testIter = testIter
        self.lastTrainId = 0
        self.lastTestId = 0