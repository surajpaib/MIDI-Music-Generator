from midi_reader import MIDIO
from midi_data_generator import DataGenerator
midi_note_parser = MIDIO("data/train")
midi_note_parser.load_notes()
datagen = DataGenerator(20)
X, y = datagen.generate_io_vectors(midi_note_parser.midi_notes)
print("Input :{} \n Output: {}".format(X, y))
trainer = MIDIRNN(X, datagen.total_notes)
trainer.initialize()
trainer.train(X, y, 100, 64)