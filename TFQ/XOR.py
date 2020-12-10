import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import random
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
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])

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
    test = sympy.symbols('test')
    c.append(cirq.rz(test).on(qubits[1]).controlled_by(qubits[0]))
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
layer1 = tfq.layers.PQC(make_circuit(qubits), readout_operators, repetitions=32, differentiator=tfq.differentiators.ParameterShift(), initializer=tf.keras.initializers.Zeros)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=layer1)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.hinge, metrics=[hinge_accuracy])

history = model.fit(train, train_label, epochs=512, batch_size=4, validation_data=(test, test_label))

print(model.trainable_weights)

plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Hinge Loss")
plt.show()


plt.plot(history.history['hinge_accuracy'], label='Training')
plt.plot(history.history['val_hinge_accuracy'], label='Validation Acc')
plt.legend()
plt.title("Training Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
