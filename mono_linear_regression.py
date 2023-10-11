from sklearn import linear_model
import numpy as np
from data_processing.preprocessing import get_training_set


class Model:
    def __init__(self, name):
        self.weight1 = np.array([])
        self.weight2 = np.array([])
        self.weight3 = np.array([])
        self.weight4 = np.array([])

    def append_weight1(self, weight):
        self.weight1 = np.append(self.weight1, weight)

    def append_weight2(self, weight):
        self.weight2 = np.append(self.weight2, weight)

    def append_weight3(self, weight):
        self.weight3 = np.append(self.weight3, weight)

    def append_weight4(self, weight):
        self.weight4 = np.append(self.weight4, weight)


def mono_linear_regression(X, T, model_instance):
    reg = linear_model.LinearRegression()
    reg.fit(X, T.T)  # Transpose T before passing it to reg.fit
    model_instance.append_weight1(reg.coef_[0, :])
    model_instance.append_weight2(reg.coef_[1, :])
    model_instance.append_weight3(reg.coef_[2, :])
    model_instance.append_weight4(reg.coef_[3, :])


if __name__ == '__main__':
    example = Model("first")
    X,T= get_training_set(0)
    print("X is:")
    print(X)

    print("T is:")
    print(T)
    mono_linear_regression(X, T, example)
    print("example.weight1:")
    print(example.weight1)
    print("example.weight2:")
    print(example.weight2)
    print("example.weight3:")
    print(example.weight3)
    print("example.weight4:")
    print(example.weight4)

