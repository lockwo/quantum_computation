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

circle_data, circle_labels = ds.make_circles(300, noise=0.2, factor=0.3, shuffle=True)
circle_data = MinMaxScaler().fit_transform(circle_data)

#plt.scatter(circle_data[circle_labels == 0][:,0], circle_data[circle_labels == 0][:,1], label='0', color='blue')
#plt.scatter(circle_data[circle_labels == 1][:,0], circle_data[circle_labels == 1][:,1], label='1', color='red')
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(circle_data, circle_labels, test_size=.2, random_state=43)
# Classical NN
nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
nn.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=3e-2), metrics=['acc'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

nn.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])
N = 400
x1, x2 = np.meshgrid(np.linspace(-0.1, 1.1, N), np.linspace(-0.1, 1.1, N))
transformed = np.c_[x1.ravel(), x2.ravel()]
z = nn(transformed)

z = np.asarray(z)
z = z.reshape(x1.shape)
cmap_light = ListedColormap(['midnightblue', 'maroon'])
plt.pcolormesh(x1, x2, z, cmap=cmap_light)
plt.scatter(circle_data[circle_labels == 0][:,0], circle_data[circle_labels == 0][:,1], label='0', color='blue')
plt.scatter(circle_data[circle_labels == 1][:,0], circle_data[circle_labels == 1][:,1], label='1', color='red')
plt.show()
plt.savefig("circle_classical")

# Quantum NN
def convert_data(data, qubits, test=False):
    cs = []
    for i in data:
        cir = cirq.Circuit()
        cir += cirq.rx(i[0] * np.pi).on(qubits[0])
        cir += cirq.rz(i[0] * np.pi).on(qubits[0])
        cir += cirq.rx(i[1] * np.pi).on(qubits[1])
        cir += cirq.rz(i[1] * np.pi).on(qubits[1])
        cs.append(cir)
    if test:
        return tfq.convert_to_tensor([cs])
    return tfq.convert_to_tensor(cs)

def encode(data, labels, qubits):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=43)
    return convert_data(X_train, qubits), convert_data(X_test, qubits), y_train, y_test

def layer(circuit, qubits, params):
    circuit += cirq.CNOT(qubits[0], qubits[1])
    circuit += cirq.ry(params[0]).on(qubits[0])
    circuit += cirq.rz(params[1]).on(qubits[0])
    circuit += cirq.ry(params[2]).on(qubits[1])
    circuit += cirq.rz(params[3]).on(qubits[1])
    return circuit

def model_circuit(qubits, depth):
    cir = cirq.Circuit()
    num_params = depth * 4
    params = sympy.symbols("q0:%d"%num_params)
    for i in range(depth):
        cir = layer(cir, qubits, params[i * 4:i * 4 + 4])
    return cir

qs = [cirq.GridQubit(0, i) for i in range(2)]
d = 10
X_train, X_test, y_train, y_test = encode(circle_data, circle_labels, qs)
c = model_circuit(qs, d)

readout_operators = [cirq.Z(qs[0])]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
#layer1 = tfq.layers.PQC(c, readout_operators, repetitions=32, differentiator=tfq.differentiators.ParameterShift())(inputs)
layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
vqc.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['acc'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

vqc.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])
N = 70
x1, x2 = np.meshgrid(np.linspace(-0.1, 1.1, N), np.linspace(-0.1, 1.1, N))
transformed = np.c_[x1.ravel(), x2.ravel()]
z = []
for i in range(len(transformed)):
    if i % 1000 == 0:
        print(i, len(transformed))
    val = convert_data([transformed[i]], qs, True)
    z.append(vqc(val).numpy()[0])

z = np.asarray(z)
z = z.reshape(x1.shape)
cmap_light = ListedColormap(['midnightblue', 'maroon'])
plt.pcolormesh(x1, x2, z, cmap=cmap_light)
plt.scatter(circle_data[circle_labels == 0][:,0], circle_data[circle_labels == 0][:,1], label='0', color='blue')
plt.scatter(circle_data[circle_labels == 1][:,0], circle_data[circle_labels == 1][:,1], label='1', color='red')
plt.savefig("circle_quantum")

