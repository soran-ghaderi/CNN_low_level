import numpy as np



def zero_pad(X, pad):

    X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), mode= 'constant', constant_values=(0,0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):

    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)

    return Z


def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev - f + (2 * pad)) / stride) + 1
    n_W = int((n_W_prev - f + (2 * pad)) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vert_start = stride * h
            vert_end = stride * h + f
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = stride * w + f
                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i,h,w,c] = conv_single_step(a_slice_prev, weights, biases)

    assert( Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)

    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            vert_start = stride * h
            vert_end = stride * h + f
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = stride * w + f
                for c in range(n_C):
                    a_prev_slice = A_prev[i, horiz_start:horiz_end, vert_start:vert_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    assert (A.shape == (m, n_H, n_W, n_C))
    return A, cache


def conv_backward(dZ, cache):

    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape

    # initialize parameters
    dA_prev = np.zeros_like(A_prev, dtype="float")
    dW = np.zeros_like(W, dtype="float")
    db = np.zeros_like(b, dtype="float")

    # padding
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start: horiz_end, :]

                    # Update gradients
                    da_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db


def distribute_value(dz, shape):

    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = [[average for h in range(n_H)] for w in range(n_W)]
    return a


x = np.random.randn(2,3)


def create_mask_from_window(x):
    """
    x: array of shape f x f
    """
    mask = x == np.max(x)
    return mask


mask = create_mask_from_window(x)
a = distribute_value(2, (2,2))


def pool_backward(dA, cache, mode="max"):

    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros_like(A_prev, dtype="float")
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start: horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]

                    elif mode == "average":

                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    assert (dA_prev.shape == A_prev.shape)
    return dA_prev