import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
import deepdish as dd
import itertools
import os


def create_model():
    # model for the neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(1024,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(15, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# import data
datah5 = dd.io.load('data/ising/ising_data_L32.h5')
#preprocessing the data
binarizer = Binarizer(threshold=0)
keys = list(datah5.keys())
for key in keys:
    datah5[key] = np.array([np.where(np.sum(slice)<0,-slice,slice) for slice in datah5[key]])
    datah5[key] = np.array([binarizer.fit_transform(slice) for slice in datah5[key]])


class_names = ['1.0','2.2', '3.0'] 



#We have to concatenate all the data and also create all the correspondent labels

data = datah5[class_names[0]]
for temperature in class_names[1:]:
    data = np.concatenate([data,datah5[temperature]])

#to create the correspondent label we just need a list [0,0,...,0,1,...,1,...]
class_labels = np.asarray(list(itertools.chain.from_iterable(itertools.repeat(x, 1000) for x in range(0,len(class_names)))))

#Split the dataset into test and train
ising_train, ising_test, temp_train, temp_test = train_test_split(data, class_labels, test_size=0.1, random_state=42)


# save checkpoints at:
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback to save our trained model every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1, period=5)

#Now we can create our neural network and print out a summary of the architecture
neural_network = create_model()
neural_network.summary()
conv = False

# reshape the data for the model
ising_train = ising_train.reshape(ising_train.shape[0],-1)
ising_test = ising_test.reshape(ising_test.shape[0], -1)

#Start training the model by testing every epoch the accuracy of the class prediction on the test set
neural_network.fit(ising_train, temp_train , epochs=100,batch_size=64,validation_data = (ising_test,temp_test),callbacks = [cp_callback])

