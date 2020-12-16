import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def make_data(qubits):
    train, train_label = [], []
    # 0 XOR 0
    cir = convert_data(qubits, cirq.Circuit(), [0, 0])
    train.append(cir)
    train_label.append(-1)
    # 1 XOR 0
    cir = convert_data(qubits, cirq.Circuit(), [1, 0])
    train.append(cir)
    train_label.append(1)
    # 0 XOR 1
    cir = convert_data(qubits, cirq.Circuit(), [0, 1])
    train.append(cir)
    train_label.append(1)
    # 1 XOR 1
    cir = convert_data(qubits, cirq.Circuit(), [1, 1])
    train.append(cir)
    train_label.append(-1)
    return tfq.convert_to_tensor(train), np.array(train_label), tfq.convert_to_tensor(train), np.array(train_label)

# X^i = Rx(pi * i)
def convert_data(qubits, cir, data):
    cir.append([cirq.X(qubits[0])**data[0]])
    cir.append([cirq.X(qubits[1])**data[1]])
    return cir

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

cmap_light = ListedColormap(['midnightblue', 'maroon'])
N = 300
x1, x2 = np.meshgrid(np.linspace(-0.1, 1.1, N), np.linspace(-0.1, 1.1, N))
transformed = np.c_[x1.ravel(), x2.ravel()]
z = []
for i in range(len(transformed)):
    val = convert_data(qubits, cirq.Circuit(), transformed[i])
    z.append(model(tfq.convert_to_tensor([val])).numpy().flatten()[0])

z = np.asarray(z)
z = z.reshape(x1.shape)
plt.pcolormesh(x1, x2, z, cmap=cmap_light)
plt.scatter([0, 1], [0, 1], color='blue', label='-1')
plt.scatter([0, 1], [1, 0], color='red', label='1')
plt.ylim(-0.1,1.1)
plt.xlim(-0.1,1.1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
