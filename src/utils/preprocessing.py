import numpy as np

def train_test_split(X, y, train_size=0.7, shuffle=True):
    """ Shuffle the data and split it into training and test data """
    # shuffle the data
    if shuffle:
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
    # split the data according to train_size
    idx_split = int(X.shape[0] * train_size)
    X_train, X_test = X[:idx_split], X[idx_split:]
    y_train, y_test = y[:idx_split], y[idx_split:]
    
    return X_train, X_test, y_train, y_test