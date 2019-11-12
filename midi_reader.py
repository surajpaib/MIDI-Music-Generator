from music21 import converter, instrument, note, chord, stream
import glob
import logging
logging.getLogger().setLevel(logging.INFO)


class MIDIO:
    def __init__(self, train_dir):
        self.midi_notes = []
        self.train_dir = train_dir

    def load_notes(self):
        for midi_file in glob.glob("{}/*.mid".format(self.train_dir)):
            logging.debug(midi_file)
            midi = converter.parse(midi_file)
            instruments = instrument.partitionByInstrument(midi)
            if instruments:
                notes_list = instruments.parts[0].recurse()
            else:
                notes_list = midi.flat.notes
            for _note in notes_list:
                if isinstance(_note, note.Note):
                    self.midi_notes.append(str(_note.pitch))
                
                elif isinstance(_note, chord.Chord):
                    self.midi_notes.append('.'.join(str(n) for n in _note.normalOrder))

    def save_midi_output(self, out, midi_out_file):
        offset = 0
        output_notes = []
        for _note in out:

            if ('.' in _note) or _note.isdigit():
                notes_in_chord = _note.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(_note)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            offset += 0.5
        midi_stream = stream.Stream(output_notes)
        try:
            midi_stream.write('midi', fp=midi_out_file)
        except:
            print("ERROR: Could not write to MIDI File.")
            

if __name__ == "__main__":
    midi_reader = MIDIO("/home/suraj/MSAI/Advanced Concepts of Machine Learning/Assignment_2_MIDI_RNN/data/maroon5")
    midi_reader.load_notes()
    print(len(midi_reader.midi_notes))