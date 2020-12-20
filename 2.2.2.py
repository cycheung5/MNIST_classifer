import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
from sklearn.metrics import accuracy_score






def knn_classifier(x_train, y_train, x_test, k):
    num_class = np.amax(y_train) + 1
    y_hat = np.zeros(len(x_test))
    #print(num_class)
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
    #print(y_hat[0])
    #print(Idxs[0,0])
    #print(y_train[373])
    return [y_hat, Idxs]


    
    



def main():
    n_array = [100, 200, 400, 600, 800, 1000]
    n_accuracies = []
    train_data = np.genfromtxt('mnist_train.csv', delimiter=',', dtype=int)
    test_data = np.genfromtxt('mnist_test.csv', delimiter=',', dtype=int)
    # 100
    x_train = train_data[1:101, 1:]
    y_train = train_data[1:101, 0]
    x_test = test_data[1:, 1:]
    y_test = test_data[1:, 0]

    # Normalize data sets by dividing them by 255
    x_train = x_train / 255
    x_test = x_test / 255

    knn = knn_classifier(x_train, y_train, x_test, 3)
    wow = accuracy_score (y_test, knn[0])
    n_accuracies.append(wow)
    



    # 200
    x_train = train_data[1:201, 1:]
    y_train = train_data[1:201, 0]
    # Normalize data sets by dividing them by 255
    x_train = x_train / 255

    knn = knn_classifier(x_train, y_train, x_test, 3)
    wow = accuracy_score (y_test, knn[0])
    n_accuracies.append(wow)
    
    # 400
    x_train = train_data[1:401, 1:]
    y_train = train_data[1:401, 0]

    # Normalize data sets by dividing them by 255
    x_train = x_train / 255

    knn = knn_classifier(x_train, y_train, x_test, 3)
    wow = accuracy_score (y_test, knn[0])
    n_accuracies.append(wow)

    # 600
    x_train = train_data[1:601, 1:]
    y_train = train_data[1:601, 0]

    # Normalize data sets by dividing them by 255
    x_train = x_train / 255

    knn = knn_classifier(x_train, y_train, x_test, 3)
    wow = accuracy_score (y_test, knn[0])
    n_accuracies.append(wow)
    
    # 800
    x_train = train_data[1:801, 1:]
    y_train = train_data[1:801, 0]

    # Normalize data sets by dividing them by 255
    x_train = x_train / 255

    knn = knn_classifier(x_train, y_train, x_test, 3)
    wow = accuracy_score (y_test, knn[0])
    n_accuracies.append(wow)
    
    # 1000
    x_train = train_data[1:, 1:]
    y_train = train_data[1:, 0]

    # Normalize data sets by dividing them by 255
    x_train = x_train / 255

    knn = knn_classifier(x_train, y_train, x_test, 3)
    wow = accuracy_score (y_test, knn[0])
    n_accuracies.append(wow)
    print(n_accuracies)

    # 2.2.2
    # Plot the accuracy score
    for i in range(len(n_array)):
        plt.scatter(n_array[i], n_accuracies[i], color='blue')
        plt.plot(n_array, n_accuracies)
        plt.xlabel('n traing set')
        plt.ylabel('Accuracy score')
        plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()
