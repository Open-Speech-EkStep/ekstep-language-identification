# ekstep-language-identification

This repository is a part of [Vakyansh's](https://open-speech-ekstep.github.io/) recipes to build state of the art Speech Recogniition Model.

The language Identification repository works for classifying the audio utterances into different classes.This repository can work for 2 or more classes depending on the requirement.

### Preparing the Data

Keep separate audio folders for different classes as well as the train and valid sets of each. The audio files should be present in .wav format.
To prepare the data edit the data paths in file data/create_manifest.py.

To run the file:
```python
python create_manifest.py
```
This creates the train and valid csv files in the ```data/ ```directory.

### Training the Model
Edit the train_config.yml file for the training parameters. Give the file path for train and valid csv's created while preparing the data.

To start the training run
```python
python train.py
```

### Inference
Edit the language_map.yml to map the labels(0,1, etc) with the languege names or codes('hi','en', etc)

To infer, edit inference.py file and provide the best_checkpoint path and audio file name.

Parameters:

model_path : Path to best_checkpoint.pt

audio_path : Audio file path

Run the file:
```python
python inference.py
```
This runs on a single audio file.
