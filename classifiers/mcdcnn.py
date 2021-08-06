# Multi Channel-Deep Convolutional Neural Network
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model


class MCDCNN:
    def __init__(self, input_shape, num_classes):
        self.shape = input_shape
        self.concat = Concatenate(axis=-1)
        self.connected = Dense(units=732, activation='relu')
        self.classifier = Dense(num_classes, activation='softmax')
        self.build()

    def build(self):
        inputs = []
        convs = []

        for _ in range(self.shape[1]):
            input_layer = Input((self.shape[0]))
            inputs.append(input_layer)

            conv1 = Conv1D(8, 5, activation='relu', padding='valid')(input_layer)
            max1 = MaxPooling1D(2)(conv1)

            conv2 = Conv1D(8, 5, activation='relu', padding='valid')(max1)
            max2 = MaxPooling1D(2)(conv2)
            flatten = Flatten()(max2)

            convs.append(flatten)

        x = self.concat(convs)
        x = self.connected(x)
        out = self.classifier(x)
        self.model = Model(inputs=inputs, outputs=out)
