from music21 import converter, instrument, note, chord
import glob
import logging
logging.getLogger().setLevel(logging.INFO)


class MIDIReader:
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


if __name__ == "__main__":
    noteacc = NoteAccumulator("data/train")
    noteacc.load_notes()
