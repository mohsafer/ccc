# ResNet
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, Dense, GlobalAveragePooling1D
from tensorflow.keras import Model


class Block(Model):
    def __init__(self, filters, kernel_size, expand=True):
        super(Block, self).__init__(name='')

        self.conv1 = Conv1D(filters, kernel_size[0], padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')

        self.conv2 = Conv1D(filters, kernel_size[1], padding='same')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

        self.conv3 = Conv1D(filters, kernel_size[2], padding='same')
        self.bn3 = BatchNormalization()

        if expand:
            self.conv4 = Conv1D(filters, kernel_size[3], padding='same')
        else:
            self.conv4 = None
        self.bn4 = BatchNormalization()

        self.add = Add()
        self.act4 = Activation('relu')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x_ = self.bn3(x)

        if self.conv4:
            x = self.conv4(x_)
            x = self.bn4(x)
        else:
            x = self.bn4(x_)

        x = self.add([x, x_])
        x = self.act4(x)

        return x


class ResNet(Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.block1 = Block(64, [8, 5, 3, 1])
        self.block2 = Block(128, [8, 5, 3, 1])
        self.block3 = Block(128, [8, 5, 3, 1], False)

        self.gap = GlobalAveragePooling1D()
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.block1(inputs)

        x = self.block2(x)

        x = self.block3(x)

        x = self.gap(x)
        return self.classifier(x)
