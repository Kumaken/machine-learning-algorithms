from MultiLayerPerceptron import MultiLayerPerceptron
from MyC45 import MyTree
from MyC45 import handleContinuousAttribute
from sklearn.utils import shuffle
import pickle
import numpy as np
import pandas as pd
import sys

def MLP(data, batch_size):

    #data = load_iris()
    # list1 = list(data.iloc[0,:-2])
    # print(list1)
    # target = list(data.iloc[0,4:])
    # print(target)
    mlp = MultiLayerPerceptron(_layers=1 , _bias=1, _inputs=4, _outputs=2, _learningRate=0.01, _maxIter=200)

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
                # print("at iteration ", i, " error = " + str(mlp.totalError))
        if j % batch_size != 0:
            mlp.updateWeight()
            mlp.updateTotalError(target)
            # print("at iteration ", i, " error = " + str(mlp.totalError))


        if(mlp.totalError < 0.01):
            break

    return mlp

def main():
    dataMLP = pd.read_csv("irisMLP.csv")
    training_data = dataMLP.sample(frac = 0.9)
    test_dataMLP = dataMLP.drop(training_data.index)
    mlp = MLP(training_data, 10)
    confusion_matrixMLP = np.zeros((3, 3))
    for i in range(len(test_dataMLP)):
        test = list(test_dataMLP.iloc[i,:-mlp.outputs])
        target = list(test_dataMLP.iloc[i,mlp.inputs:])
        estimate = mlp.estimate(test)
        # print(estimate)
        for i in range(len(estimate)):
            estimate[i] = round(estimate[i])
        # print(estimate)
        # print(target)
        row = -1
        if (target == [0, 0]):
            row = 0
        elif (target == [0, 1]):
            row = 1
        elif (target == [1, 1]):
            row = 2
        column = -1
        if (estimate == [0, 0]):
            column = 0
        elif (estimate == [0, 1]):
            column = 1
        elif (estimate == [1, 1]):
            column = 2
        if (column >= 0 and row >= 0):
            confusion_matrixMLP[row][column] += 1
    print("MLP confusion matrix:")
    print(confusion_matrixMLP)

    dataDTL = pd.read_csv("irisDTL.csv")
    dataDTL = handleContinuousAttribute(dataDTL)
    training_dataDTL = dataDTL.sample(frac = 0.9)
    test_dataDTL = dataDTL.drop(training_dataDTL.index)
    tree = MyTree(_targetAttribute = "flower")
    tree.buildTreeInit(trainingSet=training_dataDTL)
    confusion_matrixDTL = np.zeros((3, 3))
    tree.printTree()
    for i in range(len(test_dataDTL)):
        test = test_dataDTL.iloc[i]
        target = test["flower"]
        test = test.to_dict()
        for key in test:
            test[key] = str(test[key])
        # print(test)
        estimate = tree.predict(test)
        # print(estimate)
        row = -1
        row = int(target)
        column = -1
        column = int(estimate)
        if (column >= 0 and row >= 0):
            confusion_matrixDTL[row][column] += 1
    print("DTL confusion matrix:")
    print(confusion_matrixDTL)



# main()