import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def lrGradientDescent(alpha, itr):

    df = pd.DataFrame(pd.read_csv('data.csv'))  # load data
    X, y = df.X.values, df.Y.values  # input and output
    x = list(map(lambda z: (z - np.min(X))/(np.max(X) - np.min(X)), X))  # normalize the input value (X)
    m = len(X)  # number of training examples
    b0, b1 = np.random.rand(), np.random.rand()  # initialize the weights randomly
    x = np.array(x)
    y = np.array(y)
    x_avg, y_avg = np.average(x), np.average(y)
    y_hat_itr = []  # to store the predicted Y in every iterations
    r2_itr = []  # to store the r-squared value in every iterations

    '''iterate to update b0 & b1'''
    for _ in range(itr):
        yHat = b0 + b1 * x  # predicted Y
        j = yHat - y  # difference between the predicted and the actual Y
        b0 -= alpha * 2 * j.sum() / m  # update b0
        b1 -= alpha * 2 * x.dot(j).sum() / m  # update b1
        y_hat_itr_value = b0 + b1 * x
        y_hat_itr.append(y_hat_itr_value)
        r2_itr.append(1 - sum((y[i] - y_hat_itr_value[i]) ** 2 for i in range(m)) / sum((y[i] - y_avg) ** 2 for i in range(m)))

    '''calculate the predicted values'''
    y_hat = b0 + b1 * x

    '''calculate r-squared score'''
    r2 = 1 - sum((y[i] - y_hat[i]) ** 2 for i in range(m)) / sum((y[i] - y_avg) ** 2 for i in range(m))

    '''plot'''
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, marker='x', color='b', label="real value")
    plt.scatter(x, y_hat, marker='o', color="g", label="predicted value")
    plt.title("Linear Regression with Gradient Descent"), plt.legend(loc=4)
    plt.text(0, 25, "Y = {} + {}X".format(b0, b1))
    plt.text(0, 23, "r2 = {}".format(r2))
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(r2_itr)), r2_itr, marker="o", color='r', s=0.5)
    plt.xlabel("iterations"), plt.ylabel('r-squared')
    plt.show()

    return b0, b1, r2


if __name__ == "__main__":
    lrGradientDescent(0.01, 10000)
