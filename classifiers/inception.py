# InceptionTime
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense, Add, MaxPool1D, Conv1D, Activation, BatchNormalization, Concatenate
from tensorflow.keras import Model


class InceptionModule(Model):
    def __init__(self):
        super(InceptionModule, self).__init__()
        self.bottleneck = Conv1D(32, 1, strides=1, activation='relu', use_bias=False, padding='same')
        self.maxpool = MaxPool1D(3, 1, 'same')

        self.conv1 = Conv1D(32, 10, strides=1, activation='relu', use_bias=False, padding='same')
        self.conv2 = Conv1D(32, 20, strides=1, activation='relu', use_bias=False, padding='same')
        self.conv3 = Conv1D(32, 40, strides=1, activation='relu', use_bias=False, padding='same')
        self.conv4 = Conv1D(32, 1, strides=1, activation='relu', use_bias=False, padding='same')

        self.concat = Concatenate(axis=2)
        self.bn = BatchNormalization()
        self.act = Activation('relu')

    def call(self, inputs, *args, **kwargs):
        bottleneck = self.bottleneck(inputs)
        maxpool = self.maxpool(inputs)

        x1 = self.conv1(bottleneck)
        x2 = self.conv2(bottleneck)
        x3 = self.conv3(bottleneck)
        x4 = self.conv4(maxpool)

        x = self.concat([x1, x2, x3, x4])
        x = self.bn(x)
        x = self.act(x)
        return x


def shortcut(input_tensor, out_tensor):
    short = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding='same', use_bias=False)(input_tensor)
    short = BatchNormalization()(short)

    x = Add()([short, out_tensor])
    x = Activation('relu')(x)
    return x


class Inception(Model):
    def __init__(self, num_classes, depth=6):
        super(Inception, self).__init__()
        self.classifier = Dense(num_classes, activation='softmax')
        self.depth = depth
        self.gap = GlobalAveragePooling1D()

    def call(self, inputs):
        x = inputs
        x_res = inputs

        for i in range(self.depth):
            x = InceptionModule()(x)
            if i % 3 == 2:
                x = shortcut(x_res, x)
                x_res = x
        x = self.gap(x)
        return self.classifier(x)
