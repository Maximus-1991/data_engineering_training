from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Input

from src.config import *

def conv_net():
    input_layer = Input(shape=MODEL_INPUT_SHAPE)
    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='valid',
                  data_format='channels_last', activation='relu')(input_layer)
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='valid',
                  data_format='channels_last', activation='relu', strides=(2, 2))(conv)

    flatten = Flatten()(conv)
    output_layer = Dense(units=NR_CLASSES, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model