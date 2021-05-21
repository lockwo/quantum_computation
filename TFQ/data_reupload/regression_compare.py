import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import numpy as np
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from reup import ReUploadPQC

x, y = ds.load_boston(return_X_y=True)
x = MinMaxScaler().fit_transform(x)
y = (y - np.min(y)) / (np.max(y) - np.min(y))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=43)

# Quantum NN
def convert_data(data, qubits, test=False):
    cs = []
    for i in data:
        cir = cirq.Circuit()
        for j in range(len(qubits)):
            cir += cirq.rx(i[j] * np.pi).on(qubits[j])
            cir += cirq.ry(i[j] * np.pi).on(qubits[j])
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
    circuit += cirq.CNOT(qubits[-1], qubits[0])
    return circuit

def model_circuit(qubits, depth):
    cir = cirq.Circuit()
    num_params = depth * 2 * len(qubits)
    params = sympy.symbols("q0:%d"%num_params)
    for i in range(depth):
        cir = layer(cir, qubits, params[i * 2 * len(qubits):i * 2 * len(qubits) + 2 * len(qubits)])
    return cir

qs = [cirq.GridQubit(0, i) for i in range(13)]
d = 3
X_train_en, X_test_en, y_train_en, y_test_en = encode(x, y, qs)
c = model_circuit(qs, d)

readout_operators = [cirq.Z(qs[0])]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
vqc.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=3e-3))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
print(vqc.trainable_variables)
vqc_hist = vqc.fit(X_train_en, y_train_en, epochs=200, batch_size=32, validation_data=(X_test_en, y_test_en), callbacks=[callback])

lay = 8
qu = [cirq.GridQubit(0, i) for i in range(5)]
readout_operators = [cirq.Z(qu[0])]
inputs = tf.keras.layers.Input(shape=(15,))
outs = ReUploadPQC(qu, lay, readout_operators)(inputs)
vqc = tf.keras.models.Model(inputs=inputs, outputs=outs)
vqc.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=3e-3))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
print(vqc.trainable_variables)

X_train = np.pad(X_train, [(0, 0), (0, 2)])
X_test = np.pad(X_test, [(0, 0), (0, 2)])

reup_hist = vqc.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])


plt.plot(vqc_hist.history['val_loss'], label='VQC')
plt.plot(reup_hist.history['val_loss'], label='ReUp')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
plt.savefig('regress_comp')
