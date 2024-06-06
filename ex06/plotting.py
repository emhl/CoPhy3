
import sys
#sys.path.append("/Users/fdangelo/PycharmProjects/myRBM")
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import Binarizer
from my_RBM_tf2 import RBM
import deepdish as dd
from bokeh.plotting import figure
from bokeh.io import export_png

# load RBM for different temperatures
machine1 = RBM(32*32, 300, 100, (32, 32), 32,'cd')
machine1.from_saved_model('results/models/2503-141405model.h5')
machine2 = RBM(32*32, 300, 100, (32, 32), 32,'cd')
machine2.from_saved_model('results/models/2503-151600model.h5')
machine3 = RBM(32*32, 300, 100, (32, 32), 32,'cd')
machine3.from_saved_model('results/models/2503-141459model.h5')

# load data
datah5 = dd.io.load('data/ising/ising_data_L32.h5')
data_bin ={}
datah5_norm = {}
#Take spin up as standard configuration
keys = list(datah5.keys())
binarizer = Binarizer(threshold=0)
for key in keys:
    datah5_norm[key] = np.array([np.where(np.sum(slice)<0,-slice,slice) for slice in datah5[key]])
    data_bin[key] = np.array([binarizer.fit_transform(slice) for slice in datah5_norm[key]]).reshape(datah5_norm[key].shape[0],-1).astype(np.float32)

# calculate magnetization
magn_key_mean = []
for key in keys:
    print(data_bin[key].shape[0])
    magn_23 = np.array([np.mean(data_bin[key][i]) for i in range(data_bin[key].shape[0])])
    magn_key_mean.append(2*np.mean(magn_23)-1)

# load model to measure temperature
def create_model():
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

model = create_model()
model.load_weights('Training_1/cp.ckpt')
class_names = [1.0,2.2, 3.0]

# gereate new samples
steps =[200]#,1000,10000,100000,1000000]    #number of samples
for i in steps:

    fantasy_particle2,_,_,list_state1 = machine1.sample(data_bin[keys[0]][12],n_step_MC=i)
    list_magn1 = 2*np.array([np.mean(np.array(i)) for i in list_state1]) - 1
    fantasy_particle2,_,_,list_state4 = machine2.sample(data_bin[keys[1]][12],n_step_MC=i)
    list_magn4 = 2*np.array([np.mean(np.array(i)) for i in list_state4]) - 1
    fantasy_particle2,_,_,list_state5 = machine3.sample(data_bin[keys[2]][12],n_step_MC=i)
    list_magn5 = 2*np.array([np.mean(np.array(i)) for i in list_state5]) - 1

    # measure the temperature using the neural network
    temp_1 = []
    temp_4 = []
    temp_5 = []
    for a in list_state1:
        temp_1.append(class_names[np.argmax(model(np.array([a])))])
    for a in list_state4:
        temp_4.append(class_names[np.argmax(model(np.array([a])))])
    for a in list_state5:
        temp_5.append(class_names[np.argmax(model(np.array([a])))])



    # plot magnetization vs. steps
    from bokeh.models import ColumnDataSource, Whisker
    p2 = figure(title = 'Magnetization of visible states of steps in the same markov chain',plot_width=1000, plot_height=600, y_range=(0.0, 1.01))
    x = np.arange(len(list_magn1))
    # add a circle renderer with x and y coordinates, size, color, and alpha
    p2.circle(x,list_magn1, size=7, line_color="navy", fill_color="green", line_width=0.1,  fill_alpha=0.5, legend = 'Magnetization samples T=1.0' )
    p2.circle(x,list_magn4, size=7, line_color="navy", fill_color="yellow", line_width=0.1 , fill_alpha=0.5, legend = 'Magnetization samples T=2.2')
    p2.circle(x,list_magn5, size=7, line_color="navy", fill_color="red", line_width=0.1,  fill_alpha=0.5, legend = 'Magnetization samples T=3.0')

    #p2.circle(np.arange(len(magn_train)),magn_train, size=7, line_color="navy", fill_color="yellow", fill_alpha=0.5, legend = 'train')
    p2.line(x, magn_key_mean[0],line_color="green", line_width=3, line_alpha=1, legend='Magnetization data T=1.0' )
    p2.line(x, magn_key_mean[1],line_color="yellow", line_width=3, line_alpha=1, legend='Magnetization data T=2.2' )
    p2.line(x, magn_key_mean[2],line_color="red", line_width=3, line_alpha=1, legend='Magnetization data T=3.0' )
    p2.yaxis.axis_label = "Magnetization"
    p2.xaxis.axis_label = "Markov chain steps"
    p2.legend.location = "bottom_right"
    p2.legend.click_policy="hide"
    p2.output_backend = "svg"
    export_png(p2, filename=str(len(list_magn1))+"magn_step.png")


    E_1 = []
    E_4 = []
    E_5 = []
    for a in list_state1:
        E_1.append(machine1.energy(tf.reshape(a,(1024,)))[0])
    for a in list_state4:
        E_4.append(machine2.energy(tf.reshape(a,(1024,)))[0])
    for a in list_state5:
        E_5.append(machine3.energy(tf.reshape(a,(1024,)))[0])


    # plot energy vs. steps
    p3 = figure(plot_width=800, plot_height=400)
    # add a circle renderer with x and y coordinates, size, color, and alpha
    p3.line(np.arange(len(E_1)), E_1,line_color="green", line_width=2, line_alpha=0.6, legend='Energy samples T=1.0' )
    p3.line(np.arange(len(E_4)), E_4,line_color="yellow", line_width=2, line_alpha=0.6, legend='Energy samples T=2.2' )
    p3.line(np.arange(len(E_5)), E_5,line_color="red", line_width=2, line_alpha=0.6, legend='Energy samples T=3.0' )

    p3.yaxis.axis_label = "Energy"
    p3.xaxis.axis_label = "Mc step"
    p3.legend.location = "bottom_right"
    p3.output_backend = "svg"
    export_png(p3, filename=str(len(E_1))+"ener_step.png")

    #plot manetization vs. energy
    p4 = figure(plot_width=800, plot_height=400)
    # add a circle renderer with x and y coordinates, size, color, and alpha
    p4.circle(E_1,list_magn1, size=7, line_color="navy", fill_color="green", line_width=0.1,  fill_alpha=0.5, legend = 'Samples T=1.0')
    p4.circle(E_4,list_magn4, size=7, line_color="navy", fill_color="yellow", line_width=0.1 , fill_alpha=0.5, legend = 'Samples T=2.2')
    p4.circle(E_5,list_magn5, size=7, line_color="navy", fill_color="red", line_width=0.1,  fill_alpha=0.5, legend = 'Samples T=3.0')


    p4.yaxis.axis_label = "Magnetization"
    p4.xaxis.axis_label = "Energy"

    p4.legend.location = "bottom_left"
    p4.output_backend = "svg"
    export_png(p4, filename=str(len(list_magn1))+"magn_energy.png")

    # plot magnetization vs. temperature
    p5 = figure(plot_width=800, plot_height=400)
    # add a circle renderer with x and y coordinates, size, color, and alpha
    p5.circle(temp_1,list_magn1, size=7, line_color="navy", fill_color="green", line_width=0.1,  fill_alpha=0.5, legend = 'Samples T=1.0')
    p5.circle(temp_4,list_magn4, size=7, line_color="navy", fill_color="yellow", line_width=0.1 , fill_alpha=0.5, legend = 'Samples T=2.2')
    p5.circle(temp_5,list_magn5, size=7, line_color="navy", fill_color="red", line_width=0.1,  fill_alpha=0.5, legend = 'Samples T=3.0')


    p5.yaxis.axis_label = "Magnetization"
    p5.xaxis.axis_label = "Temperature"

    p5.legend.location = "bottom_left"
    p5.output_backend = "svg"
    export_png(p5, filename=str(len(list_magn1))+"magn_temp.png")

