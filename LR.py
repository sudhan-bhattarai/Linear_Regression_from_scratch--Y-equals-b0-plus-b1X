import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def lr():
    '''load data'''
    df = pd.DataFrame(pd.read_csv('data.csv'))
    x, y = df.X.values, df.Y.values

    '''calculate average'''
    x_avg, y_avg = np.average(x), np.average(y)

    '''calculate b0 and b1'''
    b1 = sum((x[i] - x_avg) * (y[i] - y_avg) for i in range(len(x))) / sum((x[i] - x_avg) ** 2 for i in range(len(x)))
    b0 = y_avg - b1 * x_avg

    '''calculate the predicted values'''
    y_hat = np.empty([len(x)])
    for i in range(len(x)):
        y_hat[i] = b0 + b1 * x[i]

    '''plot'''
    plt.scatter(x, y, marker='x', color='b', label="real value")
    plt.scatter(x, y_hat, marker='o', color="g", label="predicted value")
    plt.text(0, 25, "Y = {} + {}X".format(b0, b1))
    plt.title("Linear Regression"), plt.xlabel("X coordinate"), plt.ylabel("Y coordinate"), plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    lr()
