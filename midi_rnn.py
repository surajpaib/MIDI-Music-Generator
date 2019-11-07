from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Activation, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np


class MIDIRNN:
    def __init__(self, input_vector, output_size):
        self.input_vector = input_vector
        self.output_size = output_size


    def initialize(self):
        self.model = Sequential()
        self.model.add(CuDNNLSTM(
        512,
        input_shape=(self.input_vector.shape[1], self.input_vector.shape[2]),
        return_sequences=True
        ))
        self.model.add(Dropout(0.3))
        self.model.add(CuDNNLSTM(512, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(CuDNNLSTM(512))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.output_size))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.callbacks = [ModelCheckpoint(
            "checkpoints/weights_{epoch:02d}_loss_{loss:.4f}.hdf5", monitor='loss', 
            verbose=1,        
            save_best_only=True,        
            mode='min'
        )] 

    def train(self, X, y, iterations, batch_size):
        self.model.fit(X, y, epochs=iterations, batch_size=batch_size, callbacks=self.callbacks)
    

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


    def predict(self, X, int_to_note):
        pred_output = []
        # generate 200 notes
        for note_index in range(200):
            _X = np.reshape(X, (1, len(X), 1))
            _X = _X / float(self.output_size)
            y_pred = self.model.predict(_X, verbose=1)
            index = np.argmax(y_pred)
            result = int_to_note[index]
            pred_output.append(result)
            X.append(index)
            X = X[1:len(X)]

        return pred_output


if __name__ == "__main__":
    from midi_reader import MIDIReader
    from midi_data_generator import DataGenerator
    midi_note_parser = MIDIReader("data/train")
    midi_note_parser.load_notes()
    datagen = DataGenerator(100)
    X, y = datagen.generate_io_vectors(midi_note_parser.midi_notes)
    print("Input :{} \n Output: {}".format(X, y))
    trainer = MIDIRNN(X, datagen.total_notes)
    trainer.initialize()
    trainer.train(X, y, 100, 64)