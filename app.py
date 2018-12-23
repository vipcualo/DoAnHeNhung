from flask import Flask, request, jsonify
import numpy as np
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D,Dropout,LeakyReLU
from keras.models import Model
import tensorflow as tf
classes=['am','gsm','nfm','usb','wfm']

app = Flask(__name__)

model=None
graph=None
def load_model():
    global model
    DIM1=250
    DIM2=100
    num_classes=5
    input_signal = Input(shape=(DIM1, DIM2, 2))
    x = Conv2D(16, (3, 3), padding='same')(input_signal)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_signal, outputs=x)
    model.load_weights("/home/tainv/PycharmProjects/cnn-rtlsdr/model.h5")
    global graph
    graph = tf.get_default_graph()
    pass

def process_input(x):
    x = np.array(x)
    x = np.reshape(x, (250, 100, 2))
    x = x*1.0/255
    return np.array([x])

@app.route('/classification', methods=['POST'])
def create_task():
    url = request.json
    #sprint(url)
    x=process_input(url)
    with graph.as_default():
        res=model.predict(x)
    res=np.argmax(res)
    print(res)
    res=classes[res]
    print(res)
    return jsonify({'signal': res})

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    load_model()
    app.run(debug=False,port=5001)