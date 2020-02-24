print("Welome to Moore's law test")
print("Imports...")

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
np.reshape(inputs,(-1,1))
np.reshape(outputs,(-1,1))

print("inputs.shape = ",inputs.shape)

outputs = np.log(outputs)

'''
plt.scatter(inputs, outputs)
plt.xlabel("inputs (yesr)")
plt.ylabel("outputs")
plt.show()
'''
model = tf.keras.models.Sequential([
                        tf.keras.layers.Input(1),
                        tf.keras.layers.Dense(1) # no activation function - default is just out = wtd-sum in
                        ])

# "build the model" - I'd say "define the learning algorithm"
model.compile(error_measure?="LMS", algo?="adam")


# "train the model" - I'd say "feed the data to the previously defined topology and learning algorithm"
model.define_learning(input,output,epocchs=100)
