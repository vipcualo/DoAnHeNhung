from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D,Dropout,LeakyReLU
from keras.models import Model

def create_model(DIM1,DIM2,num_classes):
    input_signal = Input(shape=(DIM1, DIM2, 2))
    x = Conv2D(64, (3, 3), padding='same')(input_signal)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_signal, outputs=x)
    return model