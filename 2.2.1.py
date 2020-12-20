from hw3 import knn_classifier
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
from sklearn.metrics import accuracy_score




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

    knn = knn_classifier(x_train, y_train, x_test, 65)
    #print(knn[1].shape)
    print(accuracy_score (y_test, knn[0]))

    


    # 2.2.1
    # Plot the accuracy score
    accuracies = []
    k_values = [1, 3, 4, 9, 17, 33, 65]
    for k in k_values:
        y_hat_test = knn_classifier(x_train, y_train, x_test, k)
        accuracies.append(accuracy_score(y_test, y_hat_test[0]))
    
    for i in range(len(k_values)):
        plt.scatter(k_values[i], accuracies[i], color='blue')
        plt.plot(k_values, accuracies)
        plt.xlabel('K- nearest neighbors')
        plt.ylabel('Accuracy score')
        plt.tight_layout()
    plt.show()


   

if __name__ == "__main__":
    main()
