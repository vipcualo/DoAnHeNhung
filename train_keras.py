from __future__ import division, print_function
import numpy as np

from load_data import read_data
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from classification_model import create_model
from matplotlib import pyplot as plt


num_classes = 5
dim1_list = [25, 50, 100, 200]
dim2_list = [25, 50, 100, 125]
batch_size=16
epochs=2
X_train = []
Y_train = []
X_test = []
Y_test = []
data_list = ['/home/tainv/Music/test/am/txt',
             '/home/tainv/Music/test/gsm/txt',
             '/home/tainv/Music/test/nfm/txt',
             '/home/tainv/Music/test/wfm/txt'
             ]
y_list = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]


def visualize(hist,filepath):
    """Visualize training history"""
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filepath+".png")
    plt.close()


def load_data(X_train,X_test,Y_train,Y_test,path, respectively_y, test_size, factor):
    x, y = read_data(path, respectively_y, DIM1, DIM2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    for i in range(factor):
        X_train = np.concatenate((X_train, x_train), axis=0)
        X_test = np.concatenate((X_test, x_test), axis=0)
        Y_train = np.concatenate((Y_train, y_train), axis=0)
        Y_test = np.concatenate((Y_test, y_test), axis=0)
    return X_train,X_test,Y_train,Y_test

for i in range(len(dim1_list)):
    DIM1 = dim1_list[i]
    DIM2 = dim2_list[i]
    X_train = np.ones([1, DIM1, DIM2, 2])
    Y_train = np.ones([1, 4])
    X_test = np.ones([1, DIM1, DIM2, 2])
    Y_test = np.ones([1, 4])
    X_train, X_test, Y_train, Y_test=load_data(X_train,X_test,Y_train,Y_test,data_list[0], y_list[0], test_size=0.2, factor=0)
    X_train, X_test, Y_train, Y_test=load_data(X_train,X_test,Y_train,Y_test,data_list[1], y_list[1], test_size=0.2, factor=4)
    X_train, X_test, Y_train, Y_test=load_data(X_train,X_test,Y_train,Y_test,data_list[2], y_list[2], test_size=0.2, factor=4)
    X_train, X_test, Y_train, Y_test=load_data(X_train,X_test,Y_train,Y_test,data_list[3], y_list[3], test_size=0.2, factor=2)
    X_train = X_train[1:]
    X_test = X_test[1:]
    Y_train = Y_train[1:]
    Y_test = Y_test[1:]
    # ============================================== #
    filepath = "trained_model"+str(DIM1)+"-"+str(DIM2)+".h5"
    model = create_model(DIM1,DIM2,num_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
    callbacks_list = [checkpoint,early_stop]
    hist=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, shuffle=True,
              validation_data=(X_test, Y_test), class_weight='auto', verbose=2)
    visualize(hist,filepath)
