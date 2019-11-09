import numpy as np
from keras.utils import np_utils
class DataGenerator:
    def __init__(self, length):
        self.length = length
        pass
    
    def generate_io_vectors(self, midi_notes):
        self.midi_notes = midi_notes
        print(midi_notes)
        pitch_names = sorted(set(note for note in self.midi_notes))
        self.total_notes = len(set(self.midi_notes))

        note_dict = dict((note, number) for number, note in enumerate(pitch_names))
        print(note_dict)
        self.input_vector = []
        self.output_vector = []


        for i in range(0, len(self.midi_notes) - self.length):
            seq_in = self.midi_notes[i:i + self.length]
            seq_out = self.midi_notes[i+self.length]
            self.input_vector.append([note_dict[note] for note in seq_in])
            self.output_vector.append(note_dict[seq_out])
        self.input_vec = np.reshape(self.input_vector, (len(self.input_vector), self.length, 1))
        self.input_vec = self.input_vec / float(self.total_notes)
        self.output_vec = np_utils.to_categorical(self.output_vector)

        return self.input_vec, self.output_vec

    def generate_prediction_input(self):
        start = np.random.randint(0, len(self.input_vector)-1)
        pitch_names = sorted(set(note for note in self.midi_notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
        return self.input_vector[start], int_to_note

if __name__ == "__main__":
    from midi_reader import MIDIReader

    midi_note_parser = MIDIReader("data/train")
    midi_note_parser.load_notes()
    datagen = DataGenerator(100)
    X, y =datagen.generate_io_vectors(midi_note_parser.midi_notes)
    print("Input :{} \n Output: {}".format(X, y))
            


