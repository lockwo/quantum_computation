#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    s = np.pi/2
    default = circuit(weights)
    weight_copy = np.copy(weights)
    for i in range(len(weights)):
        weight_copy[i] += s
        plus = circuit(weight_copy)
        weight_copy[i] -= (2 * s)
        minus = circuit(weight_copy)
        gradient[i] = (plus - minus)/(2 * np.sin(s)) 
        hessian[i][i] = (plus - 2 * default + minus) / 2
        weight_copy[i] = weights[i]
    
    weight_copy = np.copy(weights)
    for i in range(len(hessian)):
        for j in range(i, len(hessian[i])):
            if i == j:
                continue
            else:
                weight_copy[i] += s
                weight_copy[j] += s
                plus = circuit(weight_copy)
                weight_copy[i] -= 2 * s
                minus_1 = circuit(weight_copy)
                weight_copy[i] += 2 * s
                weight_copy[j] -= 2 * s
                minus_2 = circuit(weight_copy)
                weight_copy[i] -= 2 * s
                minus_3 = circuit(weight_copy)
                hessian[i][j] = (plus - minus_1 - minus_2 + minus_3) / (2 * np.sin(s))**2
                weight_copy[i] = weights[i]
                weight_copy[j] = weights[j]
    
    for i in range(len(hessian)):
        for j in range(0, i):
            hessian[i][j] = hessian[j][i]

    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
