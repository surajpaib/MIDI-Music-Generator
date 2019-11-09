import sys
import os
import argparse
from midi_reader import MIDIO
from midi_data_generator import DataGenerator
from midi_rnn import MIDIRNN
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIDI Generation RNN')
    parser.add_argument('--sequence_length', default=20,
                    help='Length of Sequence of input notes for which one output note is predicted', type=int)
    parser.add_argument('--hidden_units', default=512,
                    help='Number of Hidden Units in the 3 layer LSTM', type=int)
    # parser.add_argument('--output_length', default=200,
    #                 help='Length of Output Sequence')                    
    parser.add_argument('--epochs', default=300,
                    help='Number of epochs', type=int)
    parser.add_argument('--batch_size', default=64,
                    help='Batch size of input', type=int)
    parser.add_argument('PATH', help= 'Path to MIDI Files')


    args = parser.parse_args()

    with open("training_args.json", "w") as fp:
        json.dump(args.__dict__, fp)

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

  
    midi_note_parser = MIDIO(args.PATH)
    midi_note_parser.load_notes()
    datagen = DataGenerator(args.sequence_length)
    X, y = datagen.generate_io_vectors(midi_note_parser.midi_notes)
    print("Input :{} \n Output: {}".format(X, y))
    trainer = MIDIRNN(X, datagen.total_notes)
    trainer.initialize(hidden_units=args.hidden_units)
    trainer.train(X, y, args.epochs, args.batch_size)