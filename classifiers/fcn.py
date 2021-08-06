# Fully Convolutional Network
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model


class FCN:
    def __init__(self, input_shape, num_classes):
        self.input = Input(input_shape)
        self.out = Dense(num_classes, activation='softmax')
        self.build()

    def build(self):
        conv1 = Conv1D(128, 8, padding='same')(self.input)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation(activation='relu')(bn1)

        conv2 = Conv1D(256, 5, padding='same')(act1)
        bn2 = BatchNormalization()(conv2)
        act2 = Activation(activation='relu')(bn2)

        conv3 = Conv1D(128, 3, padding='same')(act2)
        bn3 = BatchNormalization()(conv3)
        act3 = Activation(activation='relu')(bn3)

        gap = GlobalAveragePooling1D()(act3)

        out = self.out(gap)
        self.model = Model(inputs=self.input, outputs=out)
