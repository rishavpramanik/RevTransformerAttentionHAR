import numpy as np

def DataPreparation(data_input_file):
    data = np.load(data_input_file, allow_pickle=True)
    X = data['X']
    X = X[:, 0, :, :]
    Y = data['y']
    folds = data['folds']

    return X,Y,folds