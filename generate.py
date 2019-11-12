import argparse
from midi_reader import MIDIO
from midi_data_generator import DataGenerator
from midi_rnn import MIDIRNN
import json
import time
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIDI Generation RNN')
    parser.add_argument('--notes', default=200,
                    help='Number of notes to generate')
        parser.add_argument('--output_scheme', default="max",
                    help='Output Note to select')
    # parser.add_argument('--output_length', default=200,
    #                 help='Length of Output Sequence')                    
    parser.add_argument('WEIGHTS_PATH', 
                    help='Path to weights file')
    


    with open("training_args.json", "r") as fp:
        training_args = json.load(fp)
    args = parser.parse_args()
    midi_note_parser = MIDIO(training_args["PATH"])
    midi_note_parser.load_notes()
    datagen = DataGenerator(training_args["sequence_length"])
    X, y = datagen.generate_io_vectors(midi_note_parser.midi_notes)
    print("Input :{} \n Output: {}".format(X, y))
    trainer = MIDIRNN(X, datagen.total_notes, training_args["PATH"])

    trainer.initialize(hidden_units=training_args['hidden_units'])
    trainer.load_weights(args.WEIGHTS_PATH)
    X_test, output_map = datagen.generate_prediction_input()
    out = trainer.predict(X_test, output_map, args.notes, args.output_scheme)
    midi_note_parser.save_midi_output(out, "{}_generated.mid".format(time.time()))
