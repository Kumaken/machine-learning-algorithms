import random



class PerceptronNode():
    """A perceptron node class for multi layer perceptron

    Parameters
    ----------
    _data : int, optional
        current output of the node
    """
    def __init__(self,
                _data=0,
                _Fn=lambda x:x,
                _bias=0,
                _parents=[],
                _children=[]):
        self.data = _data
        self.Fn = _Fn # Activation Function in lambda form
        self.bias = _bias
        self.biasDelta = 0
        self.nodeDelta = 0
        self.arcDelta = []
        self.parents = _parents
        self.children = []
        self.weight = []
        for i in range(len(self.parents)):
            self.weight.append(random.uniform(0.0, 0.1))
        for i in range(len(self.weight)):
            self.arcDelta.append(0)
        for parent in self.parents:
            parent.children.append(self)

    def __str__(self):
        return super().__str__() + ", data: " + str(self.data) + ", weight: " + str(self.weight) + ", nodeDelta: " + str(self.nodeDelta) + ", arcDelta: " + str(self.arcDelta)

    def addParent(self, parent):
        self.parents.append(parent)
        parent.children.append(parent)

    def getArcWeightByParent(self, parent):
        if (len(self.parents) == 0):
            return -1
        i = 0
        while (i<len(self.parents)):
            if (parent == self.parents[i]):
                return self.weight[i]
            i+=1
        return -1

    def net(self, args):
        total = self.bias
        for i in range(len(self.weight)):
            total += args[i]*self.weight[i]
        return total

    # compute output value with activation function
    def out(self, args):
        #print(self.net(args))
        try:
            self.data = self.Fn(self.net(args))
        except OverflowError as err:
            print("Overflowed!!! Culprit -> ", args, " net: ", self.net(args))

        # print(self.data)
        return self.data

class OutputNode(PerceptronNode):
    # the catch: Has error attribute!
    def __init__(self,
                _data=0,
                _Fn=lambda x:x,
                _bias=0,
                _parents=[],
                _children=[],
                _error=0):
        super().__init__(_data=_data, _Fn=_Fn, _bias=_bias, _parents=_parents, _children=_children)
        self.error = _error

    def updateDelta(self, target, learningRate):
        # Compute weight delta per data row
        # print(target)
        self.nodeDelta = self.data * (1 - self.data) * (target - self.data)
        for i in range(len(self.weight)):
            self.arcDelta[i] += learningRate * self.nodeDelta * self.parents[i].data
        self.biasDelta += learningRate * self.nodeDelta * 1

    def updateWeight(self):
        # Compute new weight after one batch is finished
        for i in range(len(self.weight)):
            self.weight[i] += self.arcDelta[i]
            # Also reset weight delta
            self.arcDelta[i] = 0
        self.bias += self.biasDelta
        self.biasDelta = 0

    def calculateError(self, target, out):
        self.error = ((target - out)**2)/2

class HiddenNode(PerceptronNode):
    # the catch: Has error attribute!
    """

    Parameters
    ----------
    _children : array of PerceptronNode
    """

    def updateDelta(self, learningRate):
        sigma = 0
        for child in self.children:
            sigma += (child.nodeDelta * child.getArcWeightByParent(self))
            # print("sigma+=", child.nodeDelta, "*", child.getArcWeightByParent(self))
        self.nodeDelta = self.data * (1 - self.data) * sigma
        # print("nodeDelta=", self.data, "*", (1 - self.data), '*', sigma)
        for i in range(len(self.weight)):
            self.arcDelta[i] += learningRate * self.nodeDelta * self.parents[i].data
        self.biasDelta += learningRate * self.nodeDelta * 1

    def updateWeight(self):
        for i in range(len(self.weight)):
            self.weight[i] += self.arcDelta[i]
            self.arcDelta[i] = 0
        self.bias += self.biasDelta
        self.biasDelta = 0

class InputNode():
    """

    Parameters
    ----------
    _children : array of PerceptronNode
    """
    def __init__(self, _data=0, _children=[]):
        self.data = _data
        self.children = _children

    def __str__(self):
        return super().__str__() + ", data:" + str(self.data)
