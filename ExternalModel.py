import dill as pickle

def saveModel(model, filename=None):
    if filename == None:
        filename = "model.pickle"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print("Model saved")


def loadModel(filename=None):
    if filename == None:
        filename = "model.pickle"
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print("Model loaded")

    return model
