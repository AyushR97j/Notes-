# forward
z1 = W1 @ x0 + b1            # (2,)
h1 = sig(z1)                 # hidden activations
z2 = W2 @ h1 + b2            # scalar logit
yhat = sig(z2)
L = -(y0 * np.log(yhat) + (1 - y0) * np.log(1 - yhat))       # BCE
# backward
d2 = yhat - y0               # dL/dz2  (sigmoid + BCE)
gW2 = d2 * h1; gb2 = d2
d1 = (W2 * d2) * h1 * (1 - h1)                                # dL/dz1
gW1 = np.outer(d1, x0); gb1 = d1
# one SGD step
W1n, b1n, W2n, b2n = W1 - eta * gW1, b1 - eta * gb1, W2 - eta * gW2, b2 - eta * gb2
