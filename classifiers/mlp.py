# Multi-Layer perceptron
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


class MLP:
    def __init__(self, input_shape, num_classes):
        self.input = Input(input_shape)
        self.out = Dense(num_classes, activation='softmax')
        self.build()

    def build(self):
        flatten = Flatten()(self.input)

        drop1 = Dropout(0.1)(flatten)
        dense1 = Dense(500, activation='relu')(drop1)

        drop2 = Dropout(0.2)(dense1)
        dense2 = Dense(500, activation='relu')(drop2)

        drop3 = Dropout(0.2)(dense2)
        dense3 = Dense(500, activation='relu')(drop3)

        drop4 = Dropout(0.3)(dense3)
        out = self.out(drop4)
        self.model = Model(inputs=self.input, outputs=out)
