# Chord Prediction Model

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

6. Run the prediction script:

   ```bash
   TransformerPredict.py
   ```

7. In AbletonLive use the max patch `ai-harm.amxd` to play midi notes and receive their respective harmonies.

## Contributing

Contributions to this project are welcome! You can contribute by:

- Adding new features to the model architecture
- Optimizing the training process
- Enhancing the documentation

Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
