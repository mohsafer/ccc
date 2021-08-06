# Convolutional Neural Network
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Flatten
from tensorflow.keras.models import Model


class CNN:
    def __init__(self, input_shape, num_classes):
        self.input = Input(input_shape)
        self.out = Dense(units=num_classes, activation='sigmoid')
        self.build()

    def build(self):
        conv1 = Conv1D(filters=6, kernel_size=7, padding='valid', activation='sigmoid')(self.input)
        ap1 = AveragePooling1D(pool_size=3)(conv1)

        conv2 = Conv1D(filters=12, kernel_size=7, padding='valid', activation='sigmoid')(ap1)
        ap2 = AveragePooling1D(pool_size=3)(conv2)

        flatten = Flatten()(ap2)

        out = self.out(flatten)
        self.model = Model(inputs=self.input, outputs=out)
