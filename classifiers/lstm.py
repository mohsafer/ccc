from tensorflow.keras.layers import Input, Dense, LSTM, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self, input_shape, num_classes):
        self.input = Input(input_shape)
        self.out = Dense(units=num_classes, activation='sigmoid')
        self.model = self.build()

    def build(self):
        lstm1 = LSTM(units=50, activation='tanh', return_sequences=True)(self.input)
        lstm2 = LSTM(units=50, activation='tanh', return_sequences=False)(lstm1)
        flatten = Flatten()(lstm2)
        out = self.out(flatten)
        model = Model(inputs=self.input, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model
