import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq 
import sympy
import pandas as pd

def create_dense_model(N):
    x = tf.keras.layers.Input(shape=(int(2**N)))
    act = tf.keras.activations.relu
    #act = tf.keras.activations.swish
    y = tf.keras.layers.Dense(128, activation=act)(x)
    y = tf.keras.layers.Dense(256, activation=act)(y)
    y = tf.keras.layers.Dense(128, activation=act)(y)
    y = tf.keras.layers.Dense(N*3)(y)
    model = tf.keras.models.Model(inputs=x, outputs=y)
    model.summary()
    return model

def loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

def train_loss(model, y_true, y_pred):
    with tf.GradientTape() as tape:
        loss_value = tf.keras.losses.MSE(y_true, model(np.random.random_sample((1, 2)))[0][:2]-model(np.random.random_sample((1, 2)))[0][:2]+y_pred)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train(loop, model):
    losses = []
    rand_losses = []
    opt = tf.keras.optimizers.Adam()
    batch = 64
    for i in range(loop):
        expect = []
        r = []
        for j in range(batch):
            test = 10 * np.random.random_sample((1, 2**N))
            tot = 1000/np.sum(test)
            expect.append([tot * i for i in test[0]])
            rotations = model.predict(test)
            circuit = make_circuit(N, rotations[0])
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1000)
            frequencies = result.histogram(key='result', fold_func=bitstring)
            #r = [frequencies['00'], frequencies['01'], frequencies['10'], frequencies['11']]
            r.append([frequencies['0'], frequencies['1']])

        loss_value, grads = train_loss(model, expect, r)
        #print(expect, r)
        loss_value = np.average(loss_value)
        print("Epoch {}, loss {}".format(i, loss_value))
        #print(grads)
        #input()
        losses.append(loss_value)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        r = []
        for j in range(batch):
            rotations = np.random.random_sample((1, 3*N))
            circuit = make_circuit(N, rotations[0])
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1000)
            frequencies = result.histogram(key='result', fold_func=bitstring)
            r.append([frequencies['0'], frequencies['1']])
        loss_value, grads = train_loss(model, expect, r)
        loss_value = np.average(loss_value)
        rand_losses.append(loss_value)
    return losses, rand_losses

def make_circuit(N, rotations):
    c = cirq.Circuit()
    qbits = cirq.LineQubit.range(N)
    for i in range(N):
        c.append([cirq.H(qbits[i])])
    for i in range(N-1):
        c.append(cirq.CNOT(qbits[i], qbits[i+1]))
    for i in range(N):
        rx = cirq.rx(rotations[i]*3)
        c.append([rx(qbits[i])])
        ry = cirq.ry(rotations[i]*3+1)
        c.append([ry(qbits[i])])
        rz = cirq.rz(rotations[i]*3+2)
        c.append([rz(qbits[i])])
    for i in range(N):
        c.append([cirq.H(qbits[i])])
    
    c.append(cirq.measure(*qbits[:N], key='result'))
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

if __name__ == "__main__":
    N = 1 # Number of Qubits
    model = create_dense_model(N)
    l, r = train(750, model)
    plt.plot(l, label='nn')
    plt.plot(r, label='rand')
    plt.legend()
    plt.show()
    test = 10 * np.random.random_sample((1, 2**N))
    tot = 1000/np.sum(test)
    expect = [tot * i for i in test[0]]
    print(test)
    rotations = model.predict(test)
    print(rotations)
    circuit = make_circuit(N, rotations[0])
    print(circuit)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    #print(type(frequencies), frequencies['00'])
    #result = result.data.to_numpy()
    #result = result.flatten()
    #print(result)
    print('Sampled results:\n{}'.format(frequencies))
    print(expect)
