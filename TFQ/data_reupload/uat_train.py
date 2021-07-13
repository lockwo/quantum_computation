import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from uat import UAT
from scipy.special import lambertw

def create_model(lay, typ):
    inputs = tf.keras.layers.Input(shape=(1,))
    outs = UAT(lay, typ)(inputs)
    vqc = tf.keras.models.Model(inputs=inputs, outputs=outs)
    vqc.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-2))
    return vqc

def tanh(x):
    return np.tanh(5 * x)

def poly(x):
    return np.absolute(3 * np.power(x, 2) * (1 - np.power(x, 4)))

# https://arxiv.org/vc/arxiv/papers/1908/1908.08681v2.pdf
def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

fs = [tanh, poly, mish, lambertw]
lays = [1, 5, 10]

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

for layers in lays:
    for func in fs:
        uat_model = create_model(layers, "uat")
        fourier_model = create_model(layers, "fourier")

        training_data = np.random.uniform(-1, 1, (100, 1))
        training_label = func(training_data)
        validation_data = np.random.uniform(-1, 1, (100, 1))
        validation_label = func(validation_data)
        testing_data = np.random.uniform(-1.2, 1.2, (100, 1))
        plotting = np.linspace(-1.2, 1.2, 1000)

        uat_hist = uat_model.fit(training_data, training_label, epochs=200, batch_size=20, validation_data=(validation_data, validation_label), callbacks=[callback])
        fourier_hist = fourier_model.fit(training_data, training_label, epochs=200, batch_size=20, validation_data=(validation_data, validation_label), callbacks=[callback])

        plt.plot(plotting, func(plotting), color='black', label='func')
        plt.scatter(testing_data, uat_model(testing_data), label='UAT', marker='o', facecolors="None", edgecolor='red')
        plt.scatter(testing_data, fourier_model(testing_data), label='Fourier', marker='o', facecolors="None", edgecolor='blue')
        plt.legend()
        plt.show()
        save_file = "uat_" + func.__name__ + str(layers)
        plt.savefig(save_file)
        plt.clf()

