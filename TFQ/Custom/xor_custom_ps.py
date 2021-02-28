import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import matplotlib.pyplot as plt
import numpy as np

def make_data(qubits):
    train, train_label = [], []
    # 0 XOR 0
    cir = cirq.Circuit()
    cir.append([cirq.I(qubits[0])])
    cir.append([cirq.I(qubits[1])])
    train.append(cir)
    train_label.append(-1)
    # 1 XOR 0
    cir = cirq.Circuit()
    cir.append([cirq.X(qubits[0])])
    cir.append([cirq.I(qubits[1])])
    train.append(cir)
    train_label.append(1)
    # 0 XOR 1
    cir = cirq.Circuit()
    cir.append([cirq.I(qubits[0])])
    cir.append([cirq.X(qubits[1])])
    train.append(cir)
    train_label.append(1)
    # 1 XOR 1
    cir = cirq.Circuit()
    cir.append([cirq.X(qubits[0])])
    cir.append([cirq.X(qubits[1])])
    train.append(cir)
    train_label.append(-1)
    return tfq.convert_to_tensor(train), np.array(train_label), tfq.convert_to_tensor(train), np.array(train_label)

def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.rx(symbols[0]).on(bit),
        cirq.ry(symbols[1]).on(bit),
        cirq.rz(symbols[2]).on(bit))

def two_qubit_pool(source_qubit, sink_qubit, symbols):
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit

def make_circuit(qubits):
    x1 = sympy.symbols('X1_rot')
    y1 = sympy.symbols('Y1_rot')
    z1 = sympy.symbols('Z1_rot')
    x2 = sympy.symbols('X2_rot')
    y2 = sympy.symbols('Y2_rot')
    z2 = sympy.symbols('Z2_rot')
    pool = sympy.symbols('pooling0:6')
    c = cirq.Circuit()
    c.append(cirq.CNOT(qubits[0], qubits[1]))
    c.append(cirq.rx(x1).on(qubits[0]))
    c.append(cirq.ry(y1).on(qubits[0]))
    c.append(cirq.rz(z1).on(qubits[0]))
    c.append(cirq.rx(x2).on(qubits[1]))
    c.append(cirq.ry(y2).on(qubits[1]))
    c.append(cirq.rz(z2).on(qubits[1]))
    c += two_qubit_pool(qubits[0], qubits[1], pool)
    return c

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

qubits = [cirq.GridQubit(0,i) for i in range(2)]

train, train_label, test, test_label = make_data(qubits)

readout_operators = [cirq.Z(qubits[1])]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
trial_circuit = make_circuit(qubits)
print(trial_circuit)
layer1 = tfq.layers.PQC(make_circuit(qubits), readout_operators, repetitions=1000, \
    differentiator=tfq.differentiators.ParameterShift())(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=layer1)

def np_hinge(true, pred):
    t = true > 0
    p = pred > 0
    result = t == p
    return np.mean(result)

tf_loss = []
tf_acc = []
N = 100
params = np.random.uniform(0, 2 * np.pi, 12)
#params = np.zeros((12,))

model.set_weights(np.array([params]))

opt = tf.keras.optimizers.Adam(lr=0.01)

for i in range(N):
    with tf.GradientTape() as tape:
        guess = model(train)
        error = tf.keras.losses.MAE(train_label, tf.squeeze(guess))

    grad = tape.gradient(error, model.trainable_variables)
    opt.apply_gradients(zip(grad, model.trainable_variables))
    acc = np_hinge(train_label, guess.numpy().flatten())
    tf_loss.append(error)
    tf_acc.append(acc)
    if i % 10 == 0:
        print("Epoch {}/{}, Loss {}, Acc {}".format(i, N, error, acc))

import optimizers
from quantum_diffs import ParameterShift

def f(x):
    model.set_weights(np.array([x]))
    ret = model(train)
    return tf.keras.losses.MAE(train_label, tf.squeeze(ret)).numpy()

def f1(x):
    model.set_weights(np.array([x]))
    ret = model(train)
    return ret.numpy()

opt = optimizers.Adam(lr=0.01)
cutsom = []
accs = []
i = 0
while i < N:
    guess = f(params)
    cutsom.append(guess)
    gradients = ParameterShift(f, params)
    params = opt.apply_grad(gradients, params)
    acc = np_hinge(train_label, f1(params).flatten())
    accs.append(acc)
    if i % 10 == 0:
        print("Epoch {}/{}, Loss {}, Acc {}".format(i, N, guess, acc))
    i += 1

plt.plot(tf_loss, label='TFQ')
plt.plot(cutsom, label='Custom')
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MAE Loss")
plt.show()


plt.plot(tf_acc, label='TFQ')
plt.plot(accs, label='Custom')
plt.legend()
plt.title("Training Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
