## Introduction

Chord prediction is a common task in music generation and analysis. This project demonstrates the implementation of a chord prediction model using PyTorch. The model uses an Transformer-based architecture to predict chords based on sequences of input notes for real-time harmonizing. The project also contains Max patches for use in AbletonLive for realtime harmonization.

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/onurio/harmonizing-ai.git
   cd harmonizing-ai
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Put all midi files in the `midi` folder

5. Run the script to create the training data:

   ```bash
   parse-midi.py
   ```

5. Run the training notebook:

   ```bash
   HarmonizeTransformer.ipynb
   ```

## Dependencies

- Python >= 3.6
- PyTorch >= 1.5
- PyTorch Lightning >= 1.0
- Pandas

## Usage

1. Put midi files in the `/midi` folder and run the `parse_midi_full.py` script to convert them to CSV format, an `input_chords.csv` and `output_chords.csv`.
2. Run the training script using `python lstmLightning.py`.
3. Once trained, you can use the trained model for chord prediction using `python predict.py` and using the `ableton project` which uses a M4L plugin to communicate via OSC.

## Model Architecture

The chord prediction model consists of a multi-layer LSTM architecture, designed to learn the sequential patterns in the input notes and predict the corresponding chords. The model architecture is defined in the `ChordPredictionModelLightning` class in `model.py`.

## Results

After training, the model's performance can be evaluated based on various metrics such as loss and accuracy. The model's effectiveness in chord prediction can be assessed by generating chord sequences and comparing them with ground truth data.

## Contributing

Contributions to this project are welcome! You can contribute by:

- Adding new features to the model architecture
- Optimizing the training process
- Enhancing the documentation

Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
