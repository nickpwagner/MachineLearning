import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
label_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5: "dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

plt.imshow(x_train[465])
plt.title(label_dict[int(y_train[465])])
plt.show()