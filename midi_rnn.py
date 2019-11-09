from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Activation, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np



class MIDIRNN:
    def __init__(self, input_vector, output_size,train_dir):
        self.train_dir = train_dir
        self.input_vector = input_vector
        self.output_size = output_size


    def initialize(self, hidden_units):
        self.model = Sequential()
        self.model.add(CuDNNLSTM(
        512,
        input_shape=(self.input_vector.shape[1], self.input_vector.shape[2]),
        return_sequences=True
        ))
        self.model.add(Dropout(0.3))
        self.model.add(CuDNNLSTM(hidden_units, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(CuDNNLSTM(hidden_units))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.output_size))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop' , metrics=['accuracy'])
        self.callbacks = [ModelCheckpoint(
            "{}/checkpoints/weights_{epoch:02d}_loss_{loss:.4f}.hdf5".format(train_dir), monitor='loss', 
            verbose=1,        
            save_best_only=True,        
            mode='min'
        ), TensorBoard(log_dir='{}/logs'.format(train_dir), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
] 

    def train(self, X, y, iterations, batch_size):
        self.model.fit(X, y, epochs=iterations, batch_size=batch_size, callbacks=self.callbacks, validation_split=0.2)
    

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


    def predict(self, X, int_to_note, out_len):
        pred_output = []
        # generate 200 notes
        for note_index in range(out_len):
            _X = np.reshape(X, (1, len(X), 1))
            _X = _X / float(self.output_size)
            y_pred = self.model.predict(_X, verbose=1)
            index = np.argmax(y_pred)
            result = int_to_note[index]
            pred_output.append(result)
            X.append(index)
            X = X[1:len(X)]

        return pred_output