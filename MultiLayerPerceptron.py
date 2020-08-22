import numpy as np
import pandas as pd
import sys
from sklearn.utils import shuffle

from PerceptronNode import InputNode
from PerceptronNode import HiddenNode
from PerceptronNode import OutputNode

def data(o):
    return o.data

class MultiLayerPerceptron():
    def __init__(self,
                _layers = 0,
                _bias = 0,
                _inputs = 1,
                _outputs = 1,
                _learningRate = 0.1,
                _error = 0,
                _maxIter = 10,
                _Fn = lambda x: 1./(1. +
                    ( x + np.log1p(np.exp(-x) + 1) if x > 1000  else np.exp(-x)))
                ):
        super().__init__()
        self.inputs = _inputs
        self.outputs = _outputs
        self.layers = _layers #many layer
        self.inputNodes = []
        self.learningRate = _learningRate
        self.totalError = _error
        self.maxIter = _maxIter
        self.hiddenLayers = [] #contain Layer which contain HiddenNodes (list of list)
        self.outputNodes = [] #contain OutputNodes
        for i in range(self.inputs):
            self.inputNodes.append(InputNode())
        prevLayer = self.inputNodes
        for i in range(self.layers):
            layer = []
            for j in range(self.inputs):
                layer.append(HiddenNode(_Fn=_Fn, _bias=_bias, _parents=prevLayer))
            self.hiddenLayers.append(layer)
            prevLayer = layer
        for i in range(self.outputs):
            self.outputNodes.append(OutputNode(_Fn=_Fn, _bias=_bias, _parents=prevLayer))

    def __str__(self):
        result = "\n"
        result += "Input layer:\n"
        for i in range(len(self.inputNodes)):
            result += "Node " + str(i) + " " + self.inputNodes[i].__str__() + "\n"
        result += "Hidden layers:\n"
        for i in range(len(self.hiddenLayers)):
            result += "Layer " + str(i) + "\n"
            for j in range(len(self.hiddenLayers[i])):
                result += "Node " + str(j) + " " + self.hiddenLayers[i][j].__str__() + "\n"
        result += "Output layer:\n"
        for i in range(len(self.outputNodes)):
            result += "Node " + str(i) + " " + self.outputNodes[i].__str__() + "\n"
        return super().__str__() + "\n" + result

    def estimate(self, args):
        for i in range(len(self.inputNodes)):
            self.inputNodes[i].data = args[i]
        # map every inputNodes with data func! It's just a overly complicated way to make list of input nodes' data
        prevLayer = list(map(data, self.inputNodes))

        for i in range(self.layers):
            newLayer = []
            for j in range(len(self.hiddenLayers[i])):
                newLayer.append(self.hiddenLayers[i][j].out(prevLayer))
                # print(prevLayer)
            prevLayer = newLayer
        for i in range(len(self.outputNodes)):
            self.outputNodes[i].out(prevLayer)
        return list(map(data, self.outputNodes))

    def updateTotalError(self, target):
        totalError = 0
        for i in range(len(self.outputNodes)):
            self.outputNodes[i].calculateError(target[i], self.outputNodes[i].data)
            totalError += self.outputNodes[i].error
        self.totalError += totalError

    def updateDeltaWeight(self, target):
        # Update delta weight all Output Nodes
        for i in range(len(self.outputNodes)):
            self.outputNodes[i].updateDelta(target[i], self.learningRate)

        # Update delta weight all Hidden Nodes, in reverse
        i = len(self.hiddenLayers) - 1
        while i>=0:
            for j in range(len(self.hiddenLayers[i])):
                self.hiddenLayers[i][j].updateDelta(self.learningRate)
            i-=1

    def updateWeight(self):
        #update weight all Hidden Nodes
        for layer in self.hiddenLayers:
            for hiddenNode in layer:
                hiddenNode.updateWeight()

        #update weight all Output Nodes
        for node in self.outputNodes:
            node.updateWeight()
        return

def main():
    if(sys.argv[1] == None):
        sys.argv[1] = 1
    batch_size = int(sys.argv[1])

    data = pd.read_csv("irisORI.csv")
    #data = load_iris()
    # list1 = list(data.iloc[0,:-2])
    # print(list1)
    # target = list(data.iloc[0,4:])
    # print(target)
    mlp = MultiLayerPerceptron(_layers=1 , _bias=1, _inputs=4, _outputs=1, _learningRate=0.01, _maxIter=200)

    for i in range(mlp.maxIter):
        mlp.totalError = 0
        data = shuffle(data)
        #print(data.data[0])

        for j in range(len(data)):
            row = list(data.iloc[j,:-mlp.outputs])
            target = list(data.iloc[j,mlp.inputs:])
            mlp.estimate(row)
            #mlp.estimate(data.data[j])
            mlp.updateDeltaWeight(target)
            if j % batch_size == 0:
                mlp.updateWeight()
                mlp.updateTotalError(target)
                print("at iteration ", i, " error = " + str(mlp.totalError))
        if j % batch_size != 0:
            mlp.updateWeight()
            mlp.updateTotalError(target)
            print("at iteration ", i, " error = " + str(mlp.totalError))


        if(mlp.totalError < 0.01):
            break

    print(mlp)


# main()