# Encoder
from tensorflow.keras.layers import Input, Dense, PReLU, Conv1D, Dropout, Lambda, MaxPooling1D, Softmax, Multiply, Flatten
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model


class Encoder:
    def __init__(self, input_shape, num_classes):
        self.input = Input(input_shape)
        self.out = Dense(num_classes, activation='softmax')
        self.build()

    def build(self):
        # Block 1
        conv1 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(self.input)
        conv1 = InstanceNormalization()(conv1)
        conv1 = PReLU(shared_axes=[1])(conv1)
        conv1 = Dropout(rate=0.2)(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        # Block 2
        conv2 = Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)
        conv2 = InstanceNormalization()(conv2)
        conv2 = PReLU(shared_axes=[1])(conv2)
        conv2 = Dropout(rate=0.2)(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        # Block 3
        conv3 = Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
        conv3 = InstanceNormalization()(conv3)
        conv3 = PReLU(shared_axes=[1])(conv3)
        conv3 = Dropout(rate=0.2)(conv3)

        attention_data = Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = Lambda(lambda x: x[:, :, 256:])(conv3)

        attention_softmax = Softmax()(attention_softmax)
        multiply_layer = Multiply()([attention_softmax, attention_data])

        dense_layer = Dense(units=256, activation='sigmoid')(multiply_layer)
        dense_layer = InstanceNormalization()(dense_layer)

        flatten = Flatten()(dense_layer)
        out = self.out(flatten)
        self.model = Model(inputs=self.input, outputs=out)
