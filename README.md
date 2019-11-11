
# MIDI-Music-Generator

MIDI Music Generator based on code from https://github.com/Skuldur/Classical-Piano-Composer.git for an Assignment from Advanced Concepts in Machine Learning

## Install
Setup Dependencies and Fetch Code.
### Prerequisites
Python 3.x
### Instructions
`` git clone https://github.com/surajpaib/MIDI-Music-Generator.git ``
`` cd MIDI-Music-Generator ``
`` pip install -r requirements.txt``

## Running the training process
`` python train.py PATH_TO_MIDI_FILES ``

Command line parameters that can be specificed to the train.py include,

 - sequence_length
   `` python train.py PATH_TO_MIDI_FILES --sequence_length 20``
 - hidden_units
   `` python train.py PATH_TO_MIDI_FILES --hidden_units 512``
 - epochs
   `` python train.py PATH_TO_MIDI_FILES --epochs 100``
 - batch_size
    `` python train.py PATH_TO_MIDI_FILES --batch_size 64``

## Generating New Music
``python generate.py WEIGHTS_PATH``

The weights path can be found in a subdirectory within the MIDI-Music-Generator. The subdirectories are timestamped with the parameter values for the run. Inside the subdirectory both checkpoints and tensorboard log files can be found.
```
MIDI-Music-Generator
|___	1573335397.942967_30_512_128_300
	|____	checkpoints
		|______ weights.hdf5
	|____	logs
```

