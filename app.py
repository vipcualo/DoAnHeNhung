from flask import Flask, request, jsonify
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from classification_model import create_model
import tensorflow as tf
classes=['am','gsm','nfm','usb','wfm']

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=[]
)
model=None
graph=None
def load_model():
    global model
    DIM1=250
    DIM2=100
    num_classes=4
    model = create_model(DIM1,DIM2,num_classes=4)
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
@limiter.limit("60 per second")
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