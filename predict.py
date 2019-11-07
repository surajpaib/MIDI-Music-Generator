if __name__ == "__main__":
    from midi_reader import MIDIReader
    from midi_data_generator import DataGenerator
    from midi_rnn import MIDIRNN
    midi_note_parser = MIDIReader("data/train")
    midi_note_parser.load_notes()
    datagen = DataGenerator(100)
    X, y = datagen.generate_io_vectors(midi_note_parser.midi_notes)
    print("Input :{} \n Output: {}".format(X, y))
    trainer = MIDIRNN(X, datagen.total_notes)
    trainer.initialize()
    trainer.load_weights("checkpoints/weights_29_loss_1.9125.hdf5")
    X_test, output_map = datagen.generate_prediction_input()
    out = trainer.predict(X_test, output_map)
    print(out)

