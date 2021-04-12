import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import numpy as np
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

blob_data, blob_labels = ds.make_blobs(1000, centers=7, shuffle=True)
blob_data = MinMaxScaler().fit_transform(blob_data)

#plt.scatter(blob_data[blob_labels == 0][:,0], blob_data[blob_labels == 0][:,1], label='0', color='blue')
#plt.scatter(blob_data[blob_labels == 1][:,0], blob_data[blob_labels == 1][:,1], label='1', color='red')
#plt.scatter(blob_data[blob_labels == 2][:,0], blob_data[blob_labels == 2][:,1], label='2', color='green')
#plt.scatter(blob_data[blob_labels == 3][:,0], blob_data[blob_labels == 3][:,1], label='3', color='black')
#plt.scatter(blob_data[blob_labels == 4][:,0], blob_data[blob_labels == 4][:,1], label='4', color='yellow')
#plt.scatter(blob_data[blob_labels == 5][:,0], blob_data[blob_labels == 5][:,1], label='5', color='m')
#plt.scatter(blob_data[blob_labels == 6][:,0], blob_data[blob_labels == 6][:,1], label='6', color='c')
#plt.show()

# Classical NN
X_train, X_test, y_train, y_test = train_test_split(blob_data, blob_labels, test_size=.2, random_state=43)

blob_nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
blob_nn.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=3e-2), metrics=['acc'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

blob_nn.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])
cmap_light = ListedColormap(['midnightblue', 'maroon', 'forestgreen', 'dimgray', 'gold', 'violet', 'lightcyan'])
N = 400
x1, x2 = np.meshgrid(np.linspace(-0.1, 1.1, N), np.linspace(-0.1, 1.1, N))
transformed = np.c_[x1.ravel(), x2.ravel()]
z = blob_nn(transformed)

z = np.asarray(z)
z = np.argmax(z, axis=1)
z = z.reshape(x1.shape)
plt.pcolormesh(x1, x2, z, cmap=cmap_light)

plt.scatter(blob_data[blob_labels == 0][:,0], blob_data[blob_labels == 0][:,1], label='0', color='blue')
plt.scatter(blob_data[blob_labels == 1][:,0], blob_data[blob_labels == 1][:,1], label='1', color='red')
plt.scatter(blob_data[blob_labels == 2][:,0], blob_data[blob_labels == 2][:,1], label='2', color='green')
plt.scatter(blob_data[blob_labels == 3][:,0], blob_data[blob_labels == 3][:,1], label='3', color='black')
plt.scatter(blob_data[blob_labels == 4][:,0], blob_data[blob_labels == 4][:,1], label='4', color='yellow')
plt.scatter(blob_data[blob_labels == 5][:,0], blob_data[blob_labels == 5][:,1], label='5', color='m')
plt.scatter(blob_data[blob_labels == 6][:,0], blob_data[blob_labels == 6][:,1], label='6', color='c')

plt.show()
plt.savefig("blob_classical")

# Quantum NN
def convert_data(data, qubits, test=False):
    cs = []
    for i in data:
        cir = cirq.Circuit()
        for j in qubits:
            cir += cirq.rx(i[0] * np.pi).on(j)
            cir += cirq.ry(i[1] * np.pi).on(j)
        cs.append(cir)
    if test:
        return tfq.convert_to_tensor([cs])
    return tfq.convert_to_tensor(cs)

def encode(data, labels, qubits):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=43)
    return convert_data(X_train, qubits), convert_data(X_test, qubits), y_train, y_test

def layer(circuit, qubits, params):
    for i in range(len(qubits)):
        if i + 1 < len(qubits):
            circuit += cirq.CNOT(qubits[i], qubits[i + 1])
        circuit += cirq.ry(params[i * 2]).on(qubits[i])
        circuit += cirq.rz(params[i * 2 + 1]).on(qubits[i])
    return circuit

def model_circuit(qubits, depth):
    cir = cirq.Circuit()
    num_params = depth * 2 * len(qubits)
    params = sympy.symbols("q0:%d"%num_params)
    for i in range(depth):
        cir = layer(cir, qubits, params[i * 2 * len(qubits):i * 2 * len(qubits) + 2 * len(qubits)])
    return cir

qs = [cirq.GridQubit(0, i) for i in range(7)]
d = 5
X_train, X_test, y_train, y_test = encode(blob_data, blob_labels, qs)
c = model_circuit(qs, d)
print(c)

readout_operators = [cirq.Z(i) for i in qs]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
#layer1 = tfq.layers.PQC(c, readout_operators, repetitions=32, differentiator=tfq.differentiators.ParameterShift())(inputs)
layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
vqc.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['acc'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

vqc.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])
N = 60
x1, x2 = np.meshgrid(np.linspace(-0.1, 1.1, N), np.linspace(-0.1, 1.1, N))
transformed = np.c_[x1.ravel(), x2.ravel()]
z = []
for i in range(len(transformed)):
    if i % 1000 == 0:
        print(i, len(transformed))
    val = convert_data([transformed[i]], qs, True)
    z.append(vqc(val).numpy()[0])

z = np.asarray(z)
z = np.argmax(z, axis=1)
z = z.reshape(x1.shape)
cmap_light = ListedColormap(['midnightblue', 'maroon', 'forestgreen', 'dimgray', 'gold', 'violet', 'lightcyan'])
plt.pcolormesh(x1, x2, z, cmap=cmap_light)
plt.scatter(blob_data[blob_labels == 0][:,0], blob_data[blob_labels == 0][:,1], label='0', color='blue')
plt.scatter(blob_data[blob_labels == 1][:,0], blob_data[blob_labels == 1][:,1], label='1', color='red')
plt.scatter(blob_data[blob_labels == 2][:,0], blob_data[blob_labels == 2][:,1], label='2', color='green')
plt.scatter(blob_data[blob_labels == 3][:,0], blob_data[blob_labels == 3][:,1], label='3', color='black')
plt.scatter(blob_data[blob_labels == 4][:,0], blob_data[blob_labels == 4][:,1], label='4', color='yellow')
plt.scatter(blob_data[blob_labels == 5][:,0], blob_data[blob_labels == 5][:,1], label='5', color='m')
plt.scatter(blob_data[blob_labels == 6][:,0], blob_data[blob_labels == 6][:,1], label='6', color='c')
plt.show()
plt.savefig("blob_quantum")
