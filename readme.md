# Chord Prediction Model

## Introduction

Chord prediction is a common task in music generation and analysis. This project demonstrates the implementation of a chord prediction model using PyTorch. The model uses an Transformer-based architecture to predict chords based on sequences of input notes for real-time harmonizing. The project also contains Max patches for use in AbletonLive for realtime harmonization.

## examples

Melody




<audio controls>
  <source src="[https://firebasestorage.googleapis.com/v0/b/the-omri-nuri-project-website.appspot.com/o/no-harm.mp3?alt=media&token=a0f9371a-f12b-4576-8bc3-7a48c2760b96](https://github.com/onurio/harmonizing-ai/assets/36936789/f5163bc9-c3b1-48d1-8859-4973fae2a3c6
)" type="audio/mp3">
  Your browser does not support the audio element.
</audio>

Harmonized Melody

<audio controls>
  <source src="https://firebasestorage.googleapis.com/v0/b/the-omri-nuri-project-website.appspot.com/o/yes-harm.mp3?alt=media&token=df8802c7-d552-453c-be59-f4b35700f41b" type="audio/mp3">
  Your browser does not support the audio element.
</audio>


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
