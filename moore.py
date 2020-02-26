import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log

print("Welcome")

# !wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv
# I've saveed it locally

print("Read")
#moore_dataframe = pd.read_csv("moore.csv", header=None, names=["year", "Transistors"])
moore_dataframe = pd.read_csv("moore.csv", header=None)

print("Name columns")

moore_dataframe.columns  = ["year","Transistors"]

print("Scoop inputs and outputs")
inputs = np.array(moore_dataframe["year"])
outputs = np.array(moore_dataframe["Transistors"])

print("inputs.shape = ",inputs.shape)
inputs = np.reshape(inputs,(-1,1))
outputs = np.reshape(outputs,(-1,1))

print("inputs.shape = ",inputs.shape)

inputs = inputs - inputs.mean()

outputs = np.log(outputs)
print("outputs.shape = ",outputs.shape)

'''
plt.scatter(inputs, outputs)
plt.xlabel("inputs (yesr)")
plt.ylabel("outputs")
plt.show()
'''

def custom_loss(y_target,y):
    return(tf.reduce_sum(tf.abs(tf.subtract(y_target,y))))



def lr_from_epoch(epoch, lr):
    if epoch >= 50:
        return 0.0001
    else:
        return 0.001




lr_from_epoch_thingy= tf.keras.callbacks.LearningRateScheduler(lr_from_epoch)

model = tf.keras.models.Sequential([
                        tf.keras.layers.Input(1), # this shape thing appears superfluous: shape=(1,)),
                        tf.keras.layers.Dense(1) # no activation function - default is just out = wtd-sum in
                        ])

# "build the model" - I'd say "define the learning algorithm"
#model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss="mse")

learnrate_term = 0.001
momentum_term = 0.9
#model.compile(optimizer=tf.keras.optimizers.SGD(learnrate_term, momentum_term), loss=custom_loss)
model.compile(optimizer=tf.keras.optimizers.SGD(learnrate_term, momentum_term), loss="mse")

# "train the model" - I'd say "feed the data to the previously defined topology and learning algorithm"
training_callback_history = model.fit(inputs, outputs, epochs=200, callbacks=[lr_from_epoch_thingy])


weights = model.layers[0].get_weights()
print("weights type = ",type(weights))
print("len = ", len(weights))
print("weights = ", weights)

print("type(weights[0])=", type(weights[0]))
print("weights[0].shape=", weights[0].shape)
print("weights[1].shape=", weights[1].shape)

print("weights[0][0][0]=",weights[0][0][0])
print("weights[0][0,0]=",weights[0][0,0])

print("Type = ", type(training_callback_history.history["loss"]))
print("Len = ", len(training_callback_history.history["loss"]))
print(training_callback_history.history["loss"])

plt.plot(training_callback_history.history["loss"])
plt.grid()
plt.show()
