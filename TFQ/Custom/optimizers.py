import numpy as np

# Gradients and parameters must be numpy arrays

class Adam(object):
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-7) -> None:
        super().__init__()
        self.a = lr # Stepsize
        # Exponential decay rates for the moment estimates
        self.b1 = beta1
        self.b2 = beta2
        self.e = epsilon
        self.mt = 0
        self.vt = 0
        self.t = 1
        self.mhat = 0
        self.vhat = 0

    def apply_grad(self, gradients, parameters):
        self.mt = self.b1 * self.mt + (1 - self.b1) * gradients
        self.vt = self.b2 * self.vt + (1 - self.b2) * gradients**2
        self.mhat = self.mt / (1 - self.b1**self.t)
        self.vhat = self.vt / (1 - self.b2**self.t)
        parameters = parameters - self.a * self.mhat / (np.sqrt(self.vhat) + self.e)
        self.t += 1
        return parameters

class SGD(object):
    def __init__(self, lr=3e-4, m=0) -> None:
        super().__init__()
        self.a = lr
        self.momentum = m
        self.velocity = 0
        self.acceleration = 0

    def apply_grad(self, gradients, parameters):
        self.acceleration = self.momentum * self.acceleration - self.a * gradients
        self.velocity = self.velocity + self.acceleration
        parameters = parameters - self.a * gradients + self.momentum * self.velocity
        return parameters

class Newton(object):
    def __init__(self, lr=3-4) -> None:
        super().__init__()
        self.a = lr

    def apply_grad(self, gradients, hessian, parameters):
        parameters = parameters - self.a * np.linalg.pinv(hessian) @ gradients
        return parameters
