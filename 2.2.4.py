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
    train_data = np.genfromtxt('mnist_train.csv', delimiter=',', dtype=int)
    test_data = np.genfromtxt('mnist_test.csv', delimiter=',', dtype=int)
    x_train = train_data[1:, 1:]
    y_train = train_data[1:, 0]
    x_test = test_data[1:, 1:]
    y_test = test_data[1:, 0]

    # Normalize data sets by dividing them by 255
    x_train = x_train / 255
    x_test = x_test / 255

    knn = knn_classifier(x_train, y_train, x_test, 5)

    # Need to find 3 failed cases
    y_hat = knn[0]
    print(y_hat.shape)





    wrong = np.where(np.not_equal(y_hat, y_test))
    wrong = wrong[0]
    # Choose first 3
    # We get their nearest neighbors, so we need Idxs
    Idxs = knn[1]
    Idxs_1 = Idxs[wrong[0]]
    Idxs_2 = Idxs[wrong[1]]
    Idxs_3 = Idxs[wrong[3]]

   
    #print(Idxs_1)


    
    
    fig = plt.figure()
    for i in range(6):
        plt.subplot(1, 6, i +1)
        if i == 0:
            plt.imshow(np.reshape(x_test[wrong[3]], (28,28)))
            plt.axis("off")
            plt.title("Sample")
        else:
            plt.imshow(np.reshape(x_train[Idxs_3[i-1]], (28, 28)))
            plt.axis("off")
            plt.title("Neighbor " + str(i))
    plt.tight_layout()
    plt.show()

   



    
if __name__ == "__main__":
    main()
