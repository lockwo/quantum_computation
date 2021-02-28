import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import random
import matplotlib.pyplot as plt
import numpy as np

def make_data(n1, n2):
    qubit = cirq.GridQubit(0,0)
    train, test = [], []
    train_label, test_label = [], []
    for _ in range(n1):
        cir = cirq.Circuit()
        rot = random.uniform(0,0.1) if random.random() < 0.5 else random.uniform(0.9,1)
        cir.append([cirq.X(qubit)**rot])
        train.append(cir)
        if rot < 0.5:
            train_label.append(1)
        else:
            train_label.append(-1)
    for _ in range(n2):
        cir = cirq.Circuit()
        rot = random.uniform(0,0.1) if random.random() < 0.5 else random.uniform(0.9,1)
        cir.append([cirq.X(qubit)**rot])
        test.append(cir)
        if rot < 0.5:
            test_label.append(1)
        else:
            test_label.append(-1)
    return tfq.convert_to_tensor(train), np.array(train_label), tfq.convert_to_tensor(test), np.array(test_label)

def make_circuit(qubit):
    x = sympy.symbols('X_rot')
    y = sympy.symbols('Y_rot')
    z = sympy.symbols('Z_rot')
    c = cirq.Circuit()
    c.append(cirq.rx(x).on(qubit))
    c.append(cirq.ry(y).on(qubit))
    c.append(cirq.rz(z).on(qubit))
    return c

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

train, train_label, test, test_label = make_data(100, 100)

qubit = cirq.GridQubit(0,0)
readout_operators = [cirq.X(qubit)]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
trial_circuit = make_circuit(qubit)
print(trial_circuit)
layer1 = tfq.layers.PQC(make_circuit(qubit), readout_operators, repetitions=1000,\
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
#params = np.random.uniform(0, 2 * np.pi, 3)
params = np.zeros((3,))

model.set_weights(np.array([params]))

opt = tf.keras.optimizers.Adam(lr=0.01)

for i in range(N):
    with tf.GradientTape() as tape:
        guess = model(train)
        error = tf.reduce_mean(tf.keras.losses.hinge(train_label, tf.squeeze(guess)))

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
    return tf.reduce_mean(tf.keras.losses.hinge(train_label, tf.squeeze(ret))).numpy()

def f1(x):
    model.set_weights(np.array([x]))
    ret = model(train)
    return ret.numpy()

opt = optimizers.Adam(lr=0.01)
cutsom = []
accs = []

for i in range(N):
    guess = f(params)
    cutsom.append(guess)
    gradients = ParameterShift(f, params)
    params = opt.apply_grad(gradients, params)
    acc = np_hinge(train_label, f1(params).flatten())
    accs.append(acc)
    if i % 10 == 0:
        print("Epoch {}/{}, Loss {}, Acc {}".format(i, N, guess, acc))

plt.plot(tf_loss, label='TFQ')
plt.plot(cutsom, label='Custom')
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Hinge Loss")
plt.show()

plt.plot(tf_acc, label='TFQ')
plt.plot(accs, label='Custom')
plt.legend()
plt.title("Training Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
