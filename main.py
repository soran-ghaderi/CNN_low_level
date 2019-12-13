import numpy as np
import matplotlib.pyplot as plt

from nn.layers import *


def main():
    plt.rcParams["figure.figsize"] = (5.0, 4.0)
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "Accent"
    np.random.seed(1)
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)
    print("x.shape =\n", x.shape)
    print("x_pad.shape =\n", x_pad.shape)
    print("x[1,1] =\n", x[1, 1])
    print("x_pad[1,1] =\n", x_pad[1, 1])
    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])
    # plt.show()
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    hparameters = {"stride": 1, "f": 3}
    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A.shape = " + str(A.shape))
    print("A =\n", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode="average")
    print("mode = average")
    print("A.shape = " + str(A.shape))
    print("A =\n", A)
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    # Test conv_backward
    dA, dW, db = conv_backward(Z, cache_conv)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))
    np.random.seed(1)
    print('x = ', x)
    print("mask = ", mask)
    print('distributed value =', a)


if __name__ == '__main__':
    main()
