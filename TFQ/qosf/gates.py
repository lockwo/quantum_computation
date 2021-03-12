import numpy as np
import math

outer_00 = np.array([[1, 0], [0, 0]])
outer_11 = np.array([[0, 0], [0, 1]])

class X(object):
    def __init__(self, qubit=None) -> None:
        super().__init__()
        self.op = np.array([
            [0, 1],
            [1, 0]
        ])
        self.on = qubit

class Y(object):
    def __init__(self, qubit=None) -> None:
        super().__init__()
        self.op = np.array([
            [0, -1j],
            [1j, 0]
        ])
        self.on = qubit

class Z(object):
    def __init__(self, qubit=None) -> None:
        super().__init__()
        self.op = np.array([
            [1, 0],
            [0, -1]
        ])
        self.on = qubit

class H(object):
    def __init__(self, qubit=None) -> None:
        super().__init__()
        self.op = 1/math.sqrt(2) * np.array([
            [1, 1],
            [1, -1]
        ])
        self.on = qubit

class I(object):
    def __init__(self, qubit=None) -> None:
        super().__init__()
        self.op = np.array([
            [1, 0],
            [0, 1]
        ])
        self.on = qubit

class S(object):
    def __init__(self, qubit=None) -> None:
        super().__init__()
        self.op = np.array([
            [1, 0],
            [0, 1j]
        ])
        self.on = qubit

class U(object):
    def __init__(self, theta, phi, lamb, qubit=None) -> None:
        super().__init__()
        self.op = np.array([
            [math.cos(theta/2), -np.exp(1j * lamb) * math.sin(theta/2)],
            [np.exp(1j * phi) * math.sin(theta/2), np.exp(1j * (lamb + phi)) * math.cos(theta/2)]
        ])
        self.on = qubit

class Controlled(object):
    def __init__(self, op, qubits=None) -> None:
        super().__init__()
        self.on = qubits
        self.op = op.op
        self.two_qubit = True

class Rx(object):
    def __init__(self, param=0, qubit=None) -> None:
        super().__init__()
        self.param = param 
        self.op = np.array([
            [math.cos(self.param/2), -1j * math.sin(self.param/2)],
            [-1j * math.sin(param/2), math.cos(param/2)]
        ])
        self.on = qubit

class Ry(object):
    def __init__(self, param=0, qubit=None) -> None:
        super().__init__()
        self.param = param 
        self.op = np.array([
            [math.cos(self.param/2), -1 * math.sin(self.param/2)],
            [math.sin(param/2), math.cos(param/2)]
        ])
        self.on = qubit

class Rz(object):
    def __init__(self, param=0, qubit=None) -> None:
        super().__init__()
        self.param = param 
        self.op = np.array([
            [np.exp(-1j * self.param/2), 0], 
            [0, np.exp(1j * self.param/2)]
        ])
        self.on = qubit