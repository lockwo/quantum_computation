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

blob_data, blob_labels = ds.make_blobs(1000, centers=6, shuffle=True)
blob_data = MinMaxScaler().fit_transform(blob_data)

X_train, X_test, y_train, y_test = train_test_split(blob_data, blob_labels, test_size=.2, random_state=43)

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

qs = [cirq.GridQubit(0, i) for i in range(6)]
d = 5
X_train_en, X_test_en, y_train_en, y_test_en = encode(blob_data, blob_labels, qs)
c = model_circuit(qs, d)
print(c)

readout_operators = [cirq.Z(i) for i in qs]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
vqc.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['acc'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

vqc_hist = vqc.fit(X_train_en, y_train_en, epochs=100, batch_size=32, validation_data=(X_test_en, y_test_en), callbacks=[callback])

def modified_acc(y_true, y_pred):
    y_pred = tf.math.round((y_pred + 1) * 3)
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

def mod_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, (y_pred + 1) * 3)

lay = 10
qu = [cirq.GridQubit(0, 0)]
readout_operators = [cirq.Z(qu[0])]
inputs = tf.keras.layers.Input(shape=(3,))
outs = ReUploadPQC(qu, lay, readout_operators)(inputs)
vqc = tf.keras.models.Model(inputs=inputs, outputs=outs)
vqc.compile(loss=mod_loss, optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=[modified_acc])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

X_train = np.pad(X_train, [(0, 0), (0, 1)])
X_test = np.pad(X_test, [(0, 0), (0, 1)])

reup_hist = vqc.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])


plt.plot(vqc_hist.history['val_acc'], label='VQC')
plt.plot(reup_hist.history['val_modified_acc'], label='ReUp')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
plt.savefig('blob_comp')
