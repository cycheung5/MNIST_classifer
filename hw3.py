import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
from sklearn.metrics import accuracy_score






def knn_classifier(x_train, y_train, x_test, k):
    num_class = np.amax(y_train) + 1
    y_hat = np.zeros(len(x_test))
    result = distance.cdist(x_test, x_train, 'euclidean')
    sorted_matrix = np.argsort(result, axis=1)
    Idxs = sorted_matrix[:, 0:k]
    # Now need to get the majority
    
    arr = np.zeros(k)
    for row in range(len(Idxs)):
        for col in range(k):
            arr[col] = y_train[Idxs[row, col]]
        m = stats.mode(arr)
        y_hat[row] = m[0]
    return [y_hat, Idxs]


    
    
