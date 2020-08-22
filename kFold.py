import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import csv
import pandas as pd

from MyC45 import MyTree, handleContinuousAttribute, resetThresholdDict, printThresholdDict
import MultiLayerPerceptronAS

def kFold(nFold, # how many folds
         dataset, # dataset to be kFold-ed
         ):
    split = len(dataset) // nFold
    result = []
    start_idx = 0
    for i in range(nFold):
        indices = list(range(0,start_idx)) + list(range(start_idx+split,len(dataset)))
        train = dataset[indices]
        test = dataset[start_idx:start_idx+split]
        result.append( (train,test) ) # tuple of ( train dataset, test dataset)
        start_idx += split

    return result


irisDataMLP = []
with open('irisMLP.csv') as f:
    freader = csv.reader(f, delimiter=',')
    for row in freader:
        irisDataMLP.append(row)
irisDataMLP = np.array(irisDataMLP)

# print(irisDataMLP)

# kf = KFold(n_splits=10, shuffle=False, random_state=1)
attrs=['attr1', 'attr2', 'attr3', 'attr4', 'target']

nFold = 10
MLPresult = kFold(nFold, irisDataMLP)

# Evaluate for MLP:
print("Evaluating MLP Model with k-fold: ")

totalScore = 0
for train, test in MLPresult:
    print("TRAIN: --------")
    with open ('dummy.csv', mode='w', newline="") as f:
        fwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in train: # train dataset
            fwriter.writerow(row)
    # ANN:
    data = pd.read_csv("dummy.csv")
    mlp = MultiLayerPerceptronAS.callMLP(32, data)

    print("VALIDATION: --------")
    correct = 0
    for row in test: # test dataset
        values = []
        for val, attr in zip(row[:-1], attrs[:-1]):
            values.append(float(val))
        # print(values)
        prediction = float(mlp.estimate(values)[0])
        print(prediction, " vs ", row[-1])
        # threshold: 0.5
        if abs(prediction - float(row[-1])) < 0.5:
            correct+=1
    print("correct:", correct, "out of ", len(test))
    totalScore += correct / float(len(test))

print("MLP Model total score:", totalScore/nFold)

# Evaluate for C45:
irisData = []
with open('iris.csv') as f:
    freader = csv.reader(f, delimiter=',')
    for row in freader:
        irisData.append(row)
irisData = np.array(irisData)

C45result = kFold(nFold, irisData)
# print(C45result)
print("Evaluating C45 Model with k-fold: ")

totalScore = 0
# idx = 0
for train, test in C45result:
    # print(idx)
    # idx+=1
    print("TRAIN: --------")
    # trainn, testt = C45result[7]
    with open ('dummy.csv', mode='w') as f:
        fwriter = csv.writer(f)
        for row in train:
            fwriter.writerow(row)

    data = pd.read_csv("dummy.csv", header=None, names=attrs)
    training_dataset = handleContinuousAttribute(data)
    t = MyTree(_targetAttribute = 'target')
    print("=================================================")
    t.buildTreeInit(trainingSet = training_dataset)
    t.printTree()
    print("=================================================")

    print("VALIDATION: --------")
    #print(irisData[test])
    correct = 0
    for row in test:
        values = {}
        for val, attr in zip(row[:-1], attrs[:-1]):
            values[attr] = float(val)
        # print(values)
        prediction = t.predict(values)
        print(prediction, " vs ", row[-1])
        if prediction == row[-1]:
            correct+=1
    print("correct:", correct, "out of ", len(test))
    totalScore += correct / float(len(test))
    # reset threshold dict:
    resetThresholdDict()

print("MLP Model total score:", totalScore/nFold)